"""Start / resume audits on a compiled graph. Shared by the CLI, the API and the evals."""

from __future__ import annotations

from langgraph.types import Command

from .config import flush_tracing, tracing_callbacks
from .graph import run_config, summarize


async def current(app, invoice_id: str) -> dict | None:
    snap = await app.aget_state(run_config(invoice_id))
    if not snap.values:
        return None
    return summarize(snap.values, tuple(i for t in snap.tasks for i in t.interrupts))


async def start(app, invoice_id: str, restart: bool = False) -> dict:
    """Run an audit until it finishes or pauses for approval. Idempotent unless ``restart``."""
    existing = await current(app, invoice_id)
    if existing and not restart and existing["status"] != "error":
        return existing
    cfg = run_config(invoice_id, tracing_callbacks())
    try:
        await app.ainvoke({"invoice_id": invoice_id, "messages": []}, cfg)
    finally:
        flush_tracing()
    return await current(app, invoice_id)


async def decide(app, invoice_id: str, approved: bool, edits: dict | None = None) -> dict:
    state = await current(app, invoice_id)
    if not state or state["status"] != "awaiting_approval":
        raise LookupError(f"{invoice_id} is not awaiting approval (status: {state and state['status']})")
    cfg = run_config(invoice_id, tracing_callbacks())
    try:
        await app.ainvoke(Command(resume={"approved": approved, "edits": edits or {}}), cfg)
    finally:
        flush_tracing()
    return await current(app, invoice_id)
