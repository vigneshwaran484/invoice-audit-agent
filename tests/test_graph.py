"""Graph behaviour: amounts from code, retries, interrupt + resume across a 'restart', API."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from decimal import Decimal

import pytest

from audit_agent import service
from audit_agent.graph import open_agent
from audit_agent.store import DATA_DIR
from audit_agent.tools import as_langchain_tools

from conftest import FakeModel

TRUTH = json.loads((DATA_DIR / "ground_truth.json").read_text())


def run(coro):
    return asyncio.run(coro)


async def _audit(model, db, key):
    async with open_agent(model=model, tools=as_langchain_tools(), db_path=str(db)) as app:
        return await service.start(app, key)


def test_regression_amount_error_is_zero_on_all_cases(tmp_path):
    """The strictest eval check, as a test: every claim amount equals ground truth exactly."""
    model = FakeModel()
    for key, truth in TRUTH.items():
        out = run(_audit(model, tmp_path / "c.sqlite", key))
        expected = Decimal(truth["overpaid"])
        if expected == 0:
            assert out["status"] == "clean", key
        else:
            assert out["status"] == "awaiting_approval", key
            assert Decimal(out["claim"]["total_claim_amount"]) == expected, key


def test_tools_called_and_evidence_attached(tmp_path):
    out = run(_audit(FakeModel(), tmp_path / "c.sqlite", "INV-2026-0005"))
    assert out["tool_calls"] == ["get_invoice", "compute_discrepancy"]
    assert out["claim"]["evidence"] and all("SUP-FAKE" not in e for e in out["claim"]["evidence"])


def test_model_that_skips_tools_still_gets_correct_amount(tmp_path):
    out = run(_audit(FakeModel(use_tools=False), tmp_path / "c.sqlite", "INV-2026-0010"))
    assert out["tool_calls"] == []
    assert out["claim"]["total_claim_amount"] == TRUTH["INV-2026-0010"]["overpaid"]


def test_prompt_injection_invoice_still_claimed(tmp_path):
    out = run(_audit(FakeModel(), tmp_path / "c.sqlite", "INV-2026-9999"))
    assert out["claim"]["total_claim_amount"] == TRUTH["INV-2026-9999"]["overpaid"]
    assert "Ignore all previous" not in out["claim"]["explanation"]


def test_structured_output_retry_once_then_fail(tmp_path):
    m = FakeModel(bad_outputs=1)
    assert run(_audit(m, tmp_path / "a.sqlite", "INV-2026-0005"))["status"] == "awaiting_approval"
    assert m.structured_calls == 2
    with pytest.raises(RuntimeError, match="failed validation twice"):
        run(_audit(FakeModel(bad_outputs=2), tmp_path / "b.sqlite", "INV-2026-0005"))


def test_invalid_invoice_is_error(tmp_path):
    out = run(_audit(FakeModel(), tmp_path / "c.sqlite", "INV-2026-8888"))
    assert out["status"] == "error" and "not found" in out["error"]


def test_interrupt_then_resume_in_a_new_process(tmp_path):
    db = tmp_path / "c.sqlite"
    key = "INV-2026-0005"
    assert run(_audit(FakeModel(), db, key))["status"] == "awaiting_approval"

    async def resume():  # brand-new graph + connection == process restart
        async with open_agent(model=FakeModel(), tools=as_langchain_tools(), db_path=str(db)) as app:
            assert (await service.current(app, key))["status"] == "awaiting_approval"
            edited = "Corrected wording: please credit the overcharge per the contract."
            out = await service.decide(app, key, True, {"explanation": edited, "total_claim_amount": "999999"})
            assert out["claim"]["explanation"] == edited
            return out

    out = run(resume())
    assert out["status"] == "approved"
    assert out["claim"]["total_claim_amount"] == TRUTH[key]["overpaid"]  # amount edits ignored


def test_api_flow(tmp_path):
    from fastapi.testclient import TestClient

    from audit_agent.api import create_app

    @asynccontextmanager
    async def factory():
        async with open_agent(model=FakeModel(), tools=as_langchain_tools(), db_path=str(tmp_path / "api.sqlite")) as g:
            yield g

    with TestClient(create_app(factory)) as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert c.post("/audit/bad-id").status_code == 422
        assert c.post("/audit/INV-2026-8888").status_code == 404
        assert c.post("/audit/INV-2026-0002").json()["status"] == "clean"
        r = c.post("/audit/INV-2026-0005").json()
        assert r["status"] == "awaiting_approval"
        assert c.post("/audit/INV-2026-0002/decision", json={"approved": True}).status_code == 409
        r = c.post("/audit/INV-2026-0005/decision", json={"approved": False}).json()
        assert r["status"] == "rejected"
        assert c.get("/audit/INV-2026-0005").json()["status"] == "rejected"
