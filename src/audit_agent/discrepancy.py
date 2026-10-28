"""Deterministic overpayment checks.

This is the ground truth the agent is evaluated against, and the logic behind the
`compute_discrepancy` MCP tool. The LLM decides *what to look at and how to explain
it*; arithmetic stays in plain code where it can be unit-tested.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from .models import ContractTerm, Discrepancy, DiscrepancyKind, Invoice

CENT = Decimal("0.01")


def _money(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def find_term(terms: list[ContractTerm], inv: Invoice, sku: str) -> ContractTerm | None:
    for t in terms:
        if t.supplier_id == inv.supplier_id and t.sku == sku and t.valid_from <= inv.invoice_date <= t.valid_to:
            return t
    return None


def check_invoice(inv: Invoice, terms: list[ContractTerm], seen_ids: set[str] | None = None) -> list[Discrepancy]:
    """Return every discrepancy on one invoice. `seen_ids` detects duplicates across a batch."""
    out: list[Discrepancy] = []

    if seen_ids is not None:
        if inv.invoice_id in seen_ids:
            out.append(
                Discrepancy(
                    invoice_id=inv.invoice_id,
                    kind=DiscrepancyKind.DUPLICATE_INVOICE,
                    overpaid_amount=_money(inv.subtotal * (1 - inv.discount_applied_pct / 100)),
                    detail="Invoice ID already paid in this batch",
                )
            )
            return out  # the whole invoice is the overpayment; line checks would double count
        seen_ids.add(inv.invoice_id)

    for li in inv.lines:
        term = find_term(terms, inv, li.sku)
        if term is None:
            out.append(
                Discrepancy(
                    invoice_id=inv.invoice_id,
                    kind=DiscrepancyKind.NO_CONTRACT,
                    sku=li.sku,
                    overpaid_amount=Decimal("0.00"),
                    detail="No contract term covers this SKU on the invoice date",
                )
            )
            continue

        if li.unit_price > term.agreed_unit_price:
            out.append(
                Discrepancy(
                    invoice_id=inv.invoice_id,
                    kind=DiscrepancyKind.PRICE_ABOVE_CONTRACT,
                    sku=li.sku,
                    overpaid_amount=_money((li.unit_price - term.agreed_unit_price) * li.quantity),
                    detail=f"Billed {li.unit_price}/unit vs contract {term.agreed_unit_price}/unit x {li.quantity}",
                )
            )

        if term.volume_threshold and li.quantity >= term.volume_threshold and inv.discount_applied_pct < term.volume_discount_pct:
            billable = min(li.unit_price, term.agreed_unit_price) * li.quantity
            missing_pct = term.volume_discount_pct - inv.discount_applied_pct
            out.append(
                Discrepancy(
                    invoice_id=inv.invoice_id,
                    kind=DiscrepancyKind.MISSED_VOLUME_DISCOUNT,
                    sku=li.sku,
                    overpaid_amount=_money(billable * missing_pct / 100),
                    detail=f"{li.quantity} units >= {term.volume_threshold}: {term.volume_discount_pct}% due, {inv.discount_applied_pct}% applied",
                )
            )
    return out


def total_overpaid(discrepancies: list[Discrepancy]) -> Decimal:
    return _money(sum((d.overpaid_amount for d in discrepancies), Decimal("0")))
