"""Shared fixtures: a scripted chat model so graph tests run without any LLM."""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from audit_agent.graph import ClaimNarrative


class _Structured:
    def __init__(self, parent):
        self.parent = parent

    async def ainvoke(self, messages):
        self.parent.structured_calls += 1
        if self.parent.bad_outputs > 0:
            self.parent.bad_outputs -= 1
            return {"explanation": "too short"}  # fails min_length -> ValidationError
        ids = re.findall(r"\[(SUP-[A-Z]+-\d+)\]", messages[-1].content)
        return ClaimNarrative(
            explanation="Billed amounts exceed the agreed contract terms; please issue a credit note.",
            evidence_clause_ids=ids + ["SUP-FAKE-99"],  # one hallucinated ID, should be dropped
        )


class FakeModel:
    """Calls get_invoice, then compute_discrepancy, then answers. Amounts it 'claims' are wrong on purpose."""

    def __init__(self, bad_outputs: int = 0, use_tools: bool = True):
        self.bad_outputs = bad_outputs
        self.use_tools = use_tools
        self.structured_calls = 0

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return _Structured(self)

    async def ainvoke(self, messages):
        inv = re.search(r"(INV-\d{4}-\d{4}(?:#dup\d+)?)", next(m for m in messages if isinstance(m, HumanMessage)).content)[1]
        n_tool = sum(isinstance(m, ToolMessage) for m in messages)
        if self.use_tools and n_tool == 0:
            return AIMessage("", tool_calls=[{"name": "get_invoice", "args": {"invoice_id": inv}, "id": "c1"}])
        if self.use_tools and n_tool == 1:
            return AIMessage("", tool_calls=[{"name": "compute_discrepancy", "args": {"invoice_id": inv}, "id": "c2"}])
        return AIMessage("Summary: the supplier overcharged INR 1.00")  # deliberately wrong amount


@pytest.fixture
def fake_model():
    return FakeModel()
