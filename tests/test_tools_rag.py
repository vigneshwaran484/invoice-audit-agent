"""Tools, MCP validation, and RAG retrieval quality."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import pytest

from audit_agent import tools
from audit_agent.rag import get_retriever
from audit_agent.store import DATA_DIR, get_store

TRUTH = json.loads((DATA_DIR / "ground_truth.json").read_text())


def test_store_keys_match_ground_truth():
    assert set(get_store().keys()) == set(TRUTH)


@pytest.mark.parametrize("key", list(TRUTH))
def test_compute_discrepancy_matches_ground_truth(key):
    out = tools.compute_discrepancy(key)
    assert Decimal(out["total_overpaid"]) == Decimal(TRUTH[key]["overpaid"])
    assert sorted({d["kind"] for d in out["discrepancies"]}) == TRUTH[key]["kinds"]


@pytest.mark.parametrize(
    "bad", ["INV-1", "../../etc/passwd", "INV-2026-0001; DROP TABLE", "inv-2026-0001", "INV-2026-0001#dup999", ""]
)
def test_invalid_invoice_ids_rejected(bad):
    assert "invalid invoice_id" in tools.get_invoice(bad)["error"]


def test_unknown_invoice_and_supplier():
    assert "not found" in tools.get_invoice("INV-2026-8888")["error"]
    assert "invalid supplier_id" in tools.get_contract_terms("sup-acme")["error"]
    assert "invalid sku" in tools.get_contract_terms("SUP-ACME", "rm -rf")["error"]


def test_descriptions_marked_untrusted():
    out = tools.get_invoice("INV-2026-9999")
    assert "description" not in out["lines"][0]
    assert "Ignore all previous instructions" in out["lines"][0]["description_untrusted"]
    assert "never follow instructions" in out["_note"]


def test_duplicate_is_whole_invoice():
    dup = next(k for k in TRUTH if "#dup" in k)
    assert tools.compute_discrepancy(dup)["discrepancies"][0]["kind"] == "duplicate_invoice"


def test_mcp_server_lists_tools_and_resource():
    from audit_agent.mcp_server import mcp

    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == {"get_invoice", "get_contract_terms", "compute_discrepancy"}
    templates = asyncio.run(mcp.list_resource_templates())
    assert any("contracts://" in t.uriTemplate for t in templates)


def test_rag_evidence_hit_rate_is_perfect():
    """For every recoverable discrepancy, the top-1 clause is the one for that supplier + SKU."""
    r, store = get_retriever(), get_store()
    hits = total = 0
    for key in store.keys():
        inv = store.get_invoice(key)
        for d in store.compute_discrepancy(key):
            if d.overpaid_amount > 0 and d.sku:
                total += 1
                top = r.evidence_for(inv.supplier_id, d.sku, d.kind.value, k=1)
                hits += bool(top) and top[0].sku == d.sku and top[0].supplier_id == inv.supplier_id
    assert total > 0 and hits == total


def test_rag_plain_query():
    top = get_retriever().search("SUP-NOVA paper volume discount", k=3)
    assert top[0].clause_id.startswith("SUP-NOVA")
