"""Domain models. These are also the structured-output schemas the LLM must fill."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class LineItem(BaseModel):
    sku: str
    description: str
    quantity: int = Field(gt=0)
    unit_price: Decimal = Field(gt=0, decimal_places=2)

    @property
    def total(self) -> Decimal:
        return self.unit_price * self.quantity


class Invoice(BaseModel):
    invoice_id: str
    supplier_id: str
    invoice_date: date
    lines: list[LineItem] = Field(min_length=1)
    discount_applied_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)

    @property
    def subtotal(self) -> Decimal:
        return sum((li.total for li in self.lines), Decimal("0"))


class ContractTerm(BaseModel):
    """Negotiated price for one SKU, plus a volume discount, valid in a date window."""

    supplier_id: str
    sku: str
    agreed_unit_price: Decimal = Field(gt=0, decimal_places=2)
    volume_threshold: int | None = Field(default=None, gt=0, description="Units per invoice that unlock the discount")
    volume_discount_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    valid_from: date
    valid_to: date

    @field_validator("valid_to")
    @classmethod
    def _window(cls, v: date, info):
        start = info.data.get("valid_from")
        if start and v < start:
            raise ValueError("valid_to must be on or after valid_from")
        return v


class DiscrepancyKind(str, Enum):
    PRICE_ABOVE_CONTRACT = "price_above_contract"
    MISSED_VOLUME_DISCOUNT = "missed_volume_discount"
    DUPLICATE_INVOICE = "duplicate_invoice"
    NO_CONTRACT = "no_contract"  # not an overpayment by itself: flag for a human


class Discrepancy(BaseModel):
    invoice_id: str
    kind: DiscrepancyKind
    sku: str | None = None
    overpaid_amount: Decimal = Field(ge=0, decimal_places=2)
    detail: str


class Claim(BaseModel):
    """What the agent drafts for a human to approve (structured output target)."""

    invoice_id: str
    supplier_id: str
    total_claim_amount: Decimal = Field(ge=0, decimal_places=2)
    discrepancies: list[Discrepancy]
    explanation: str = Field(min_length=20, description="Plain-English reason a supplier's AP team would accept")
    evidence: list[str] = Field(default_factory=list, description="Contract clause snippets or IDs supporting the claim")
