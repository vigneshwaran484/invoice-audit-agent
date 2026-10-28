"""Day 4 · MCP server exposing the audit tools (FastMCP, stdio transport).

Tools (all READ-ONLY; nothing here may write or pay):
  - get_invoice(invoice_id)               -> Invoice as dict, or {"error": ...}
  - get_contract_terms(supplier_id, sku?) -> contract terms for that supplier
  - compute_discrepancy(invoice_id)       -> discrepancies via discrepancy.check_invoice
Resource:
  - contracts://{supplier_id}             -> the clause text, for the RAG step
Prompt:
  - audit_invoice(invoice_id)             -> the standard audit instruction

Security (see store.py / tools.py):
  - Every argument is validated (^INV-\\d{4}-\\d{4}(#dupN)?$, ^SUP-[A-Z]+$, ^SKU-...$); anything else
    is rejected before any lookup. No file paths or SQL are built from tool input.
  - Return data, not instructions: invoice descriptions are returned as ``description_untrusted``
    with a note. INV-2026-9999 is a poisoned invoice used to show the agent ignores injected text.
  - Result sizes are capped (MAX_ITEMS / MAX_TEXT).

Run it:  uv run python -m audit_agent.mcp_server
Inspect: npx @modelcontextprotocol/inspector uv run python -m audit_agent.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import tools
from .store import ToolInputError, get_store

mcp = FastMCP("invoice-audit")

for _f in tools.TOOL_FUNCS:
    mcp.tool()(_f)


@mcp.resource("contracts://{supplier_id}")
def contract_clauses(supplier_id: str) -> str:
    """Contract clause text for one supplier."""
    try:
        return get_store().clause_text(supplier_id)
    except ToolInputError as e:
        return f"error: {e}"


@mcp.prompt()
def audit_invoice(invoice_id: str) -> str:
    """Standard instruction for auditing one invoice."""
    return (
        f"Audit invoice {invoice_id}. Call get_invoice, then compute_discrepancy. "
        "Report only amounts returned by compute_discrepancy."
    )


def main() -> None:
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
