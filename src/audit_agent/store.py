"""Read-only access to the synthetic dataset, with strict argument validation.

Shared by the MCP server and the in-process LangChain tools, so both enforce the same rules:
  - every ID is validated against a strict regex before it touches any lookup,
  - nothing is ever written,
  - results are size-capped.

Duplicate invoices share an ``invoice_id``. They are addressed with the same keys as
``ground_truth.json``: ``INV-2026-0007`` is the first submission, ``INV-2026-0007#dup1`` the second.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from .discrepancy import check_invoice
from .models import ContractTerm, Discrepancy, Invoice

DATA_DIR = Path(os.environ.get("AUDIT_DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "out"))

INVOICE_KEY_RE = re.compile(r"^INV-\d{4}-\d{4}(#dup\d{1,2})?$")
SUPPLIER_RE = re.compile(r"^SUP-[A-Z]{1,20}$")
SKU_RE = re.compile(r"^SKU-[A-Z0-9-]{1,20}$")
MAX_ITEMS = 50  # cap so one tool call cannot flood the context window
MAX_TEXT = 4000


class ToolInputError(ValueError):
    """Raised for invalid or unknown tool arguments. The message is safe to show to the model."""


def validate(value: str, pattern: re.Pattern[str], what: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value.strip()):
        raise ToolInputError(f"invalid {what}: must match {pattern.pattern}")
    return value.strip()


class Store:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.terms = [ContractTerm.model_validate(t) for t in self._load("contracts.json")]
        invoices = [Invoice.model_validate(i) for i in self._load("invoices.json")]
        self.clauses_md = (self.data_dir / "contract_clauses.md").read_text()

        # Key every submission the same way ground_truth.json does, and precompute the
        # batch-aware discrepancies (duplicates need the order of submission).
        self.invoices: dict[str, Invoice] = {}
        self._discrepancies: dict[str, list[Discrepancy]] = {}
        seen: set[str] = set()
        counts: dict[str, int] = {}
        for inv in invoices:
            n = counts[inv.invoice_id] = counts.get(inv.invoice_id, 0) + 1
            key = inv.invoice_id if n == 1 else f"{inv.invoice_id}#dup{n - 1}"
            self.invoices[key] = inv
            self._discrepancies[key] = check_invoice(inv, self.terms, seen)

    def _load(self, name: str):
        path = self.data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run: uv run python data/generate.py")
        return json.loads(path.read_text())

    # ---- read-only queries ----
    def keys(self) -> list[str]:
        return list(self.invoices)

    def get_invoice(self, invoice_key: str) -> Invoice:
        key = validate(invoice_key, INVOICE_KEY_RE, "invoice_id")
        if key not in self.invoices:
            raise ToolInputError(f"invoice {key} not found")
        return self.invoices[key]

    def get_contract_terms(self, supplier_id: str, sku: str | None = None) -> list[ContractTerm]:
        sup = validate(supplier_id, SUPPLIER_RE, "supplier_id")
        if sku is not None:
            sku = validate(sku, SKU_RE, "sku")
        out = [t for t in self.terms if t.supplier_id == sup and (sku is None or t.sku == sku)]
        return out[:MAX_ITEMS]

    def compute_discrepancy(self, invoice_key: str) -> list[Discrepancy]:
        self.get_invoice(invoice_key)  # validates + not-found
        return list(self._discrepancies[invoice_key.strip()])[:MAX_ITEMS]

    def clause_text(self, supplier_id: str) -> str:
        sup = validate(supplier_id, SUPPLIER_RE, "supplier_id")
        marker = f"## Master supply agreement: {sup}\n"
        if marker not in self.clauses_md:
            raise ToolInputError(f"no contract for {sup}")
        section = self.clauses_md.split(marker, 1)[1].split("\n## ", 1)[0]
        return (marker + section)[:MAX_TEXT]


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()
