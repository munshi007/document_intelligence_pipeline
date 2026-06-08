"""Behavioural tests for the structural parameter-dedup pass in
ExtractorAgent._synthesize_results (the Category-C de-hardcoding of the old
`field == 'parameters'` literal).

These exercise the REAL method (with a stubbed VLM client) rather than a
re-implementation, and lock in the two invariants that matter:

  1. Parameter-shaped lists (items exposing name/value/unit) are deduped by
     name and get value/unit splitting — regardless of the field's name.
  2. Non-parameter-shaped lists (e.g. line_items) are never deduped or mutated,
     so duplicate-description invoice rows survive.
"""
from typing import List, Optional

from pydantic import BaseModel

from extractor.agent import ExtractorAgent


class TechParameter(BaseModel):
    name: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    total: Optional[float] = None


class RecordWithParameters(BaseModel):
    parameters: List[TechParameter] = []
    line_items: List[LineItem] = []
    reasoning_thoughts: Optional[str] = None


class RecordWithTechParams(BaseModel):
    technical_parameters: List[TechParameter] = []
    reasoning_thoughts: Optional[str] = None


class _NullClient:
    """Stub VLM client: makes the final distillation step a no-op so the test
    isolates the dedup pass (generate_structured -> None hits the fallback)."""

    def generate_structured(self, *args, **kwargs):
        return None

    def generate(self, *args, **kwargs):
        return ""


def _agent() -> ExtractorAgent:
    agent = ExtractorAgent.__new__(ExtractorAgent)  # bypass __init__ (no model load)
    agent.client = _NullClient()
    return agent


def test_parameters_dedup_and_unit_split_keep_line_items():
    agent = _agent()
    rec = RecordWithParameters(
        parameters=[
            TechParameter(name="Voltage", value="24 V", unit=None),  # unit should split out
            TechParameter(name="Voltage", value="24", unit="V"),     # duplicate name -> dropped
            TechParameter(name="Current", value="3.5", unit="A"),
        ],
        line_items=[
            LineItem(description="Widget", quantity=1.0, total=10.0),
            LineItem(description="Widget", quantity=2.0, total=20.0),  # dup description -> MUST survive
        ],
    )

    out = agent._synthesize_results([rec], RecordWithParameters, domain="General")

    assert len(out.parameters) == 2, "duplicate-named parameter should be dropped"
    assert out.parameters[0].value == "24" and out.parameters[0].unit == "V", "unit split failed"
    assert len(out.line_items) == 2, "line_items must never be deduped"
    assert out.line_items[0].description == out.line_items[1].description == "Widget"


def test_differently_named_param_field_also_dedups():
    agent = _agent()
    rec = RecordWithTechParams(
        technical_parameters=[
            TechParameter(name="Temp", value="70", unit="C"),
            TechParameter(name="Temp", value="70", unit="C"),  # duplicate -> dropped
        ],
    )

    out = agent._synthesize_results([rec], RecordWithTechParams, domain="General")

    assert len(out.technical_parameters) == 1, (
        "param-shaped list under a non-'parameters' name should still dedup"
    )
