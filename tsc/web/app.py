"""FastAPI web application for TSC v2.0."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tsc.config import LLMProvider, settings
from tsc.llm.factory import create_llm_client
from tsc.pipeline.orchestrator import TSCPipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="TSC v2.0", description="Feature Evaluation Pipeline")

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path("/tmp/tsc_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


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
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

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

        if provider:
            settings.llm_provider = LLMProvider(provider)
        if model:
            settings.llm_model = model

        # Setup pipeline
        pipeline = TSCPipeline()

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
