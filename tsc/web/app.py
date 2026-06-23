from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
import sys
import multiprocessing

# -- CRITICAL MACOS FIX --
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from unittest.mock import MagicMock
sys.modules['grpc'] = MagicMock()
sys.modules['grpc.experimental'] = MagicMock()
sys.modules['grpc.experimental.aio'] = MagicMock()
sys.modules['grpc_status'] = MagicMock()
sys.modules['grpc_status.rpc_status'] = MagicMock()
# ------------------------

# --- PREWARM PYTORCH BEFORE ASYNCIO EVENT LOOP ---
try:
    from tsc.memory.world_rag import _get_embedder, _get_reranker
    _get_embedder()
    _get_reranker()
except Exception:
    pass

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

from tsc.config import LLMProvider, Settings, settings
from tsc.llm.factory import create_llm_client
from tsc.pipeline.orchestrator import TSCPipeline
from tsc.api.persona_api import router as persona_router  # G6

logger = logging.getLogger(__name__)

app = FastAPI(title="TSC v2.0", description="Feature Evaluation Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# G6: mount persona REST API
app.include_router(persona_router)

DIST_DIR = Path("/Users/mohnish/Downloads/tsc architecture/predictive_ui/dist")
UPLOAD_DIR = Path("/tmp/tsc_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── G8: RFC 7807 Problem Details error handlers ──────────────────────────────

def _problem(status: int, title: str, detail: str, instance: str = "") -> JSONResponse:
    """Return an application/problem+json response per RFC 7807."""
    return JSONResponse(
        status_code=status,
        content={"type": f"https://tsc.api/errors/{title.lower().replace(' ', '-')}",
                 "title": title, "status": status,
                 "detail": detail, "instance": instance},
        headers={"Content-Type": "application/problem+json"},
    )


@app.exception_handler(StarletteHTTPException)
async def http_problem_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _problem(exc.status_code, exc.detail or "HTTP Error",
                    str(exc.detail), str(request.url.path))


@app.exception_handler(RequestValidationError)
async def validation_problem_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{"field": ".".join(str(l) for l in e["loc"]), "message": e["msg"]}
               for e in exc.errors()]
    return _problem(422, "Validation Error",
                    "One or more request fields are invalid.",
                    str(request.url.path))


# ── Static Files ─────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(DIST_DIR / "index.html")


@app.get("/favicon.svg")
async def serve_favicon():
    return FileResponse(DIST_DIR / "favicon.svg")


@app.get("/icons.svg")
async def serve_icons():
    return FileResponse(DIST_DIR / "icons.svg")


@app.get("/boardroom_shot.png")
async def serve_boardroom_shot():
    return FileResponse(DIST_DIR / "boardroom_shot.png")


@app.get("/oasis_shot.png")
async def serve_oasis_shot():
    return FileResponse(DIST_DIR / "oasis_shot.png")


@app.get("/combined_dashboard.png")
async def serve_combined_dashboard():
    return FileResponse(DIST_DIR / "combined_dashboard.png")


@app.get("/predictive_dashboard.png")
async def serve_predictive_dashboard():
    return FileResponse(DIST_DIR / "predictive_dashboard.png")


app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")
app.mount("/models", StaticFiles(directory=str(DIST_DIR / "models")), name="models")


# ── REST Endpoints ───────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_files(
    interviews: Optional[UploadFile] = File(None),
    support: Optional[UploadFile] = File(None),
    analytics: Optional[UploadFile] = File(None),
    context: Optional[UploadFile] = File(None),
    proposal: Optional[UploadFile] = File(None),
):
    """Upload input files for evaluation."""
    files = {}
    for name, upload in [
        ("interviews", interviews),
        ("support", support),
        ("analytics", analytics),
        ("context", context),
        ("proposal", proposal),
    ]:
        if upload and upload.filename:
            path = UPLOAD_DIR / upload.filename
            content = await upload.read()
            path.write_bytes(content)
            files[name] = str(path)

    if not files:
        return JSONResponse(
            {"error": "No files uploaded"}, status_code=400
        )

    return {"files": files, "message": f"{len(files)} files uploaded"}


class TextUploadPayload(BaseModel):
    feature_proposal: str
    company_context: str
    support_tickets: str
    customer_interviews: str
    analytics: Optional[str] = "{}"

@app.post("/api/upload_text")
async def upload_text(payload: TextUploadPayload):
    """Save raw text payloads as files for evaluation."""
    import time
    timestamp = int(time.time())
    
    files = {}
    
    # Save Feature Proposal
    proposal_path = UPLOAD_DIR / f"proposal_{timestamp}.json"
    proposal_path.write_text(json.dumps({"text": payload.feature_proposal}))
    files["proposal"] = str(proposal_path)
    
    # Save Company Context
    context_path = UPLOAD_DIR / f"context_{timestamp}.json"
    context_path.write_text(json.dumps({"text": payload.company_context}))
    files["context"] = str(context_path)
    
    # Save Support Tickets
    support_path = UPLOAD_DIR / f"support_{timestamp}.txt"
    support_path.write_text(payload.support_tickets)
    files["support"] = str(support_path)
    
    # Save Customer Interviews
    interviews_path = UPLOAD_DIR / f"interviews_{timestamp}.txt"
    interviews_path.write_text(payload.customer_interviews)
    files["interviews"] = str(interviews_path)
    
    # Save Analytics
    analytics_path = UPLOAD_DIR / f"analytics_{timestamp}.txt"
    analytics_path.write_text(payload.analytics)
    if payload.analytics.strip():
        files["analytics"] = str(analytics_path)
    
    return {"files": files, "message": "Text files saved"}


@app.get("/api/status")
async def get_status():
    return {
        "status": "ready",
        "version": "2.0.0",
        "provider": settings.llm_provider.value,
        "model": settings.llm_model,
    }


class CommandPayload(BaseModel):
    action: str
    type: Optional[str] = None
    data: Optional[dict] = None
    target_agent_id: Optional[str] = None
    questions: Optional[list[str]] = None
    event: Optional[str] = None

class RefineSeedsPayload(BaseModel):
    seeds: list[str]
    instruction: str
    provider: Optional[str] = None
    model: Optional[str] = None

@app.post("/api/simulation/refine_seeds")
async def refine_seeds(payload: RefineSeedsPayload):
    """Refine seed posts using the specified LLM."""
    try:
        from tsc.config import LLMProvider, Settings, settings
        from tsc.llm.factory import create_llm_client
        import json
        
        req_settings = Settings(**{k: v for k, v in settings.model_dump().items()})
        if payload.provider:
            req_settings.llm_provider = LLMProvider(payload.provider)
        if payload.model:
            req_settings.llm_model = payload.model
            
        llm = create_llm_client(settings=req_settings)
        
        system_prompt = (
            "## 1. Identity & Role\n"
            "You are an expert community manager and content strategist specializing in social media and platform engagement.\n\n"
            
            "## 2. Capabilities & Constraints\n"
            "- You can rewrite, refine, or adjust the tone of social media posts based on user instructions.\n"
            "- You must strictly adhere to the user's specific skill instructions, persona requests, or constraints.\n"
            "- You must NOT hallucinate facts not present in the original posts unless instructed to do so.\n"
            "- You must return exactly the same number of posts as provided.\n\n"
            
            "## 3. Behavioral Guidelines\n"
            "- Apply the requested tone and instruction seamlessly to all provided posts.\n"
            "- Ensure the final posts feel authentic, human, and match the structural format of the originals.\n\n"
            
            "## 4. Output Format\n"
            "You must return ONLY a valid JSON array of strings. Do not include markdown formatting, explanations, or conversational text.\n"
            "[\n"
            '  "post 1 content...",\n'
            '  "post 2 content..."\n'
            "]"
        )
        
        user_prompt = (
            "<current_seed_posts>\n"
            f"{json.dumps(payload.seeds, indent=2)}\n"
            "</current_seed_posts>\n\n"
            "<instruction>\n"
            f"{payload.instruction}\n"
            "</instruction>\n\n"
            "Based on the instruction, rewrite the seed posts. Return ONLY the JSON array."
        )
        
        schema = {
            "type": "array",
            "items": {"type": "string"}
        }
        
        res = await llm.analyze(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=schema,
            temperature=0.7
        )
        
        return {"status": "success", "seeds": res}
    except Exception as e:
        logger.error(f"Failed to refine seeds: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulation/{run_id}/command")
async def send_simulation_command(run_id: str, payload: CommandPayload):
    """Write an IPC command to the simulation's active directory."""
    from fastapi import HTTPException
    
    import os
    
    # Locate the run directory
    search_paths = []
    
    env_dir = os.environ.get("OASIS_RUNS_DIR")
    if env_dir:
        search_paths.append(Path(env_dir) / run_id)
        
    search_paths.extend([
        Path(f"log/oasis_runs/{run_id}"),
        Path(f"/Users/mohnish/Downloads/tsc architecture/log/oasis_runs/{run_id}"),
        Path(f"/tmp/oasis_runs/{run_id}")
    ])
    
    run_dir = None
    for p in search_paths:
        if p.exists() and p.is_dir():
            run_dir = p
            break
            
    if not run_dir:
        raise HTTPException(status_code=404, detail=f"Simulation run directory '{run_id}' not found.")
        
    command_file = run_dir / "commands.json"
    
    try:
        command_file.write_text(payload.model_dump_json())
        return {"status": "success", "message": "Command sent successfully."}
    except Exception as e:
        logger.error(f"Failed to write command file: {e}")
        raise HTTPException(status_code=500, detail="Failed to send command to simulation engine.")


# ── WebSocket for Real-Time Evaluation ───────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.connections: set[WebSocket] = set()  # G10: set not list — O(1) disconnect

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.add(ws)

    def disconnect(self, ws: WebSocket):
        self.connections.discard(ws)  # discard is safe if not present

    async def send_json(self, ws: WebSocket, data: dict):
        try:
            await ws.send_json(data)
        except Exception:
            self.disconnect(ws)

manager = ConnectionManager()

# Holds the currently running pipeline task so /api/simulation/stop can cancel it.
# None when no simulation is active.
_active_pipeline_task: Optional[asyncio.Task] = None
_active_pipeline: Optional[TSCPipeline] = None


@app.websocket("/ws/evaluate")
async def ws_evaluate(ws: WebSocket):
    """Run evaluation with real-time progress via WebSocket."""
    global _active_pipeline_task, _active_pipeline
    await manager.connect(ws)
    try:
        # Receive config
        config = await ws.receive_json()
        files = config.get("files", {})
        provider = config.get("provider")
        model = config.get("model")

        print("[DEBUG] ws_evaluate: Received config")
        # G11: build a per-request config copy — never mutate global settings
        req_settings = Settings(
            **{k: v for k, v in settings.model_dump().items()}
        )
        if provider:
            req_settings.llm_provider = LLMProvider(provider)
        if model:
            req_settings.llm_model = model

        print("[DEBUG] ws_evaluate: Initializing TSCPipeline...")
        # Setup pipeline with per-request config
        pipeline = TSCPipeline(cfg=req_settings)
        print("[DEBUG] ws_evaluate: TSCPipeline initialized successfully!")

        async def on_progress(layer, name, status, details):
            await manager.send_json(ws, {
                "type": "progress",
                "layer": layer,
                "name": name,
                "status": status,
                "details": details,
            })

        pipeline.set_progress_callback(
            lambda l, n, s, d: asyncio.ensure_future(on_progress(l, n, s, d))
        )

        # Rely on orchestrator's default interactive callback which writes to pipeline.jsonl and polls commands.json
        # pipeline.set_interactive_callback(on_interactive)

        # Run — register this task so /api/simulation/stop can cancel it
        _active_pipeline_task = asyncio.current_task()
        _active_pipeline = pipeline
        await manager.send_json(ws, {"type": "started"})
        boardroom_only = config.get("boardroom_only", False)
        result = await pipeline.evaluate(**files, boardroom_only=boardroom_only)

        await manager.send_json(ws, {
            "type": "complete",
            "result": json.loads(result.model_dump_json()),
        })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except asyncio.CancelledError:
        # User hit Stop — write a stop event so the port-8080 bridge can notify the frontend
        logger.info("Pipeline cancelled by user request")
        try:
            pipeline._write_jsonl_event({"type": "simulation_stopped", "reason": "user_requested"})
        except Exception:
            pass
        raise  # re-raise so asyncio properly marks the task as cancelled
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        await manager.send_json(ws, {
            "type": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        })
    finally:
        _active_pipeline_task = None
        _active_pipeline = None
        manager.disconnect(ws)


@app.post("/api/simulation/stop")
async def stop_simulation():
    """Gracefully signal the simulation engine to stop and aggregate results."""
    global _active_pipeline_task, _active_pipeline
    
    if _active_pipeline and hasattr(_active_pipeline, '_pipeline_jsonl') and _active_pipeline._pipeline_jsonl:
        run_dir = _active_pipeline._pipeline_jsonl.parent
        commands_file = run_dir / "commands.json"
        
        try:
            import json
            with open(commands_file, "w") as f:
                json.dump({"action": "stop"}, f)
            return {"status": "stopping", "message": "Stop command sent to OASIS engine"}
        except Exception as e:
            logger.error(f"Failed to write stop command: {e}")
            return {"status": "error", "message": f"Failed to write stop command: {e}"}
            
    # Fallback to hard cancel if pipeline is stuck elsewhere
    if _active_pipeline_task and not _active_pipeline_task.done():
        _active_pipeline_task.cancel()
        return {"status": "stopping", "message": "Pipeline hard cancellation requested"}
        
    return {"status": "idle", "message": "No simulation currently running"}


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the web server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
