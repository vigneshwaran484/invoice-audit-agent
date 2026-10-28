"""Day 8 · FastAPI service.

  POST /audit/{invoice_id}          -> starts or continues a thread; returns the draft claim or "clean"
  POST /audit/{invoice_id}/decision -> {"approved": bool, "edits": {...}} resumes the interrupted graph
  GET  /audit/{invoice_id}          -> current state of the thread
  GET  /health

Run: uv run uvicorn audit_agent.api:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import service
from .graph import open_agent
from .store import INVOICE_KEY_RE


class Decision(BaseModel):
    approved: bool
    edits: dict = {}


def create_app(agent_factory=open_agent) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with agent_factory() as graph:
            app.state.graph = graph
            yield

    app = FastAPI(title="Invoice Audit Agent", lifespan=lifespan)

    def _check(invoice_id: str) -> None:
        if not INVOICE_KEY_RE.fullmatch(invoice_id):
            raise HTTPException(422, "invalid invoice_id")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/audit/{invoice_id}")
    async def audit(invoice_id: str, restart: bool = False):
        _check(invoice_id)
        result = await service.start(app.state.graph, invoice_id, restart=restart)
        if result["status"] == "error":
            raise HTTPException(404, result["error"])
        return result

    @app.get("/audit/{invoice_id}")
    async def get_audit(invoice_id: str):
        _check(invoice_id)
        result = await service.current(app.state.graph, invoice_id)
        if result is None:
            raise HTTPException(404, "no audit for this invoice yet")
        return result

    @app.post("/audit/{invoice_id}/decision")
    async def decision(invoice_id: str, body: Decision):
        _check(invoice_id)
        try:
            return await service.decide(app.state.graph, invoice_id, body.approved, body.edits)
        except LookupError as e:
            raise HTTPException(409, str(e)) from e

    return app


app = create_app()
