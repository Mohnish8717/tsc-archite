from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

from tsc.config import LLMProvider, Settings, settings
from tsc.llm.factory import create_llm_client
from tsc.pipeline.orchestrator import TSCPipeline
from tsc.api.persona_api import router as persona_router  # G6

logger = logging.getLogger(__name__)

app = FastAPI(title="TSC v2.0", description="Feature Evaluation Pipeline")

# G6: mount persona REST API
app.include_router(persona_router)

STATIC_DIR = Path(__file__).parent / "static"
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
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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


@app.websocket("/ws/evaluate")
async def ws_evaluate(ws: WebSocket):
    """Run evaluation with real-time progress via WebSocket."""
    await manager.connect(ws)
    try:
        # Receive config
        config = await ws.receive_json()
        files = config.get("files", {})
        provider = config.get("provider")
        model = config.get("model")

        # G11: build a per-request config copy — never mutate global settings
        req_settings = Settings(
            **{k: v for k, v in settings.model_dump().items()}
        )
        if provider:
            req_settings.llm_provider = LLMProvider(provider)
        if model:
            req_settings.llm_model = model

        # Setup pipeline with per-request config
        pipeline = TSCPipeline(cfg=req_settings)

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

        # Run
        await manager.send_json(ws, {"type": "started"})
        result = await pipeline.evaluate(**files)

        await manager.send_json(ws, {
            "type": "complete",
            "result": json.loads(result.model_dump_json()),
        })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        await manager.send_json(ws, {
            "type": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        })
    finally:
        manager.disconnect(ws)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the web server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
