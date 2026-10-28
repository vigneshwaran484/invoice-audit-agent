from datetime import date
from decimal import Decimal

from audit_agent.discrepancy import check_invoice, total_overpaid
from audit_agent.models import ContractTerm, DiscrepancyKind, Invoice, LineItem

TERM = ContractTerm(
    supplier_id="SUP-X",
    sku="SKU-1",
    agreed_unit_price=Decimal("100.00"),
    volume_threshold=50,
    volume_discount_pct=Decimal("5"),
    valid_from=date(2026, 1, 1),
    valid_to=date(2026, 12, 31),
)


def inv(qty=10, price="100.00", discount="0", day=date(2026, 6, 1), sku="SKU-1", iid="INV-1"):
    return Invoice(
        invoice_id=iid,
        supplier_id="SUP-X",
        invoice_date=day,
        lines=[LineItem(sku=sku, description="x", quantity=qty, unit_price=Decimal(price))],
        discount_applied_pct=Decimal(discount),
    )


def kinds(found):
    return {d.kind for d in found}


def test_clean_invoice_has_no_discrepancies():
    assert check_invoice(inv(), [TERM]) == []


def test_price_above_contract_is_charged_per_unit():
    found = check_invoice(inv(qty=10, price="112.50"), [TERM])
    assert kinds(found) == {DiscrepancyKind.PRICE_ABOVE_CONTRACT}
    assert total_overpaid(found) == Decimal("125.00")


def test_missed_volume_discount():
    found = check_invoice(inv(qty=60), [TERM])
    assert kinds(found) == {DiscrepancyKind.MISSED_VOLUME_DISCOUNT}
    assert total_overpaid(found) == Decimal("300.00")  # 5% of 6000


def test_discount_applied_correctly_is_not_flagged():
    assert check_invoice(inv(qty=60, discount="5"), [TERM]) == []


def test_overprice_and_missed_discount_do_not_double_count_the_discount_base():
    found = check_invoice(inv(qty=60, price="110.00"), [TERM])
    assert kinds(found) == {DiscrepancyKind.PRICE_ABOVE_CONTRACT, DiscrepancyKind.MISSED_VOLUME_DISCOUNT}
    # 600 overprice + 5% of the *contract* price base (6000), not of the inflated 6600
    assert total_overpaid(found) == Decimal("900.00")


def test_contract_outside_validity_window_counts_as_no_contract():
    found = check_invoice(inv(day=date(2027, 2, 1)), [TERM])
    assert kinds(found) == {DiscrepancyKind.NO_CONTRACT}
    assert total_overpaid(found) == Decimal("0.00")


def test_duplicate_invoice_is_whole_amount_and_skips_line_checks():
    seen: set[str] = set()
    assert check_invoice(inv(qty=10, price="112.50"), [TERM], seen)
    dup = check_invoice(inv(qty=10, price="112.50"), [TERM], seen)
    assert kinds(dup) == {DiscrepancyKind.DUPLICATE_INVOICE}
    assert total_overpaid(dup) == Decimal("1125.00")
