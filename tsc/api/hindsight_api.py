from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/hindsight")


class HindsightQueryPayload(BaseModel):
    bank_id: str
    query: str
    max_tokens: int = 800


@router.post("/query")
async def query_hindsight_bank(payload: HindsightQueryPayload):
    """Query any Hindsight bank by raw bank_id and a free-text question.

    bank_id examples:
      - "boardroom-CEO_Jane"          (boardroom agent bank)
      - "oasis-run-1750000000"        (OASIS unified bank: agent traces + focus groups)
    """
    if not payload.bank_id.strip():
        raise HTTPException(status_code=400, detail="bank_id must not be empty.")
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty.")

    # Create a lightweight session instance. __init__ connects immediately via HINDSIGHT_URL.
    # We do NOT call initialize() — no run_id needed, we query by raw bank_id.
    from tsc.memory.hindsight_session import HindsightSessionManager
    session = HindsightSessionManager()

    if not session.is_connected and not payload.bank_id.startswith("world-"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Hindsight server is not reachable. "
                "Ensure HINDSIGHT_URL is set and the Hindsight Docker container is running."
            ),
        )

    try:
        if payload.bank_id.startswith("world-"):
            from tsc.memory.world_rag import WorldRAGEngine
            from tsc.web.app import _active_pipeline
            
            run_id = payload.bank_id.split("world-", 1)[1]
            
            if _active_pipeline and _active_pipeline.world_rag:
                active_run_id = _active_pipeline.simulation_id
                if active_run_id != run_id:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Cannot query past run '{run_id}' while simulation '{active_run_id}' "
                            "is actively running. Wait for it to finish."
                        )
                    )
            
            engine = WorldRAGEngine()
            await engine.initialize(run_id)
            results = await engine.query(payload.query, run_id=run_id, top_k=5)
            answer = "\\n\\n".join([r.text for r in results])
        else:
            answer = await session.raw_recall(
                bank_id=payload.bank_id,
                query=payload.query,
                max_tokens=payload.max_tokens,
            )
    except Exception as e:
        logger.error(f"Hindsight query failed for bank '{payload.bank_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Hindsight query error: {e}")

    if not answer or not answer.strip():
        answer = (
            f"No relevant data found in bank '{payload.bank_id}' for this query. "
            "The bank may be empty, or the data has not yet been indexed."
        )

    # max_tokens * 4 approximates max characters (1 token ≈ 4 chars)
    return {
        "answer": answer[: payload.max_tokens * 4],
        "bank_id": payload.bank_id,
        "query": payload.query,
    }
