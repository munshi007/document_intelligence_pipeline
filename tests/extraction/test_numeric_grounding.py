"""Behavioural tests for numeric grounding in the span verifier (task #51).

Strings were verified against the source; numbers were not — live runs showed
2243.0 extracted for a 243,00 € cell, 1100.0 for 1.000,00 €, 11.8 for 1,80,
all sailing through with pass_rate 1.0. The verifier now checks every numeric
leaf (locale-aware) against its own table row first, then the whole document,
snaps unique near-misses to the row value, and flags the rest.

All three fixtures below are the exact errors observed live on
sample-invoice (runs of 2026-06-10).
"""
from extractor.agent import ExtractorAgent

SOURCE = """<!-- page:1 -->
| Service Description | Amount -without VAT- | quantity | Total Amount |
| --- | --- | --- | --- |
| Basic Fee wmView | 130,00 € | 1 | 130,00 € |
| Basic Fee wmGuide | 1.000,00 € | 0 | 0,00 € |
| Transaction Fee T1 | 0,58 € | 14 | 8,12 € |
| Transaction Fee T3 | 1,50 € | 162 | 243,00 € |
| Transaction Fee T6 | 1,80 € | 0 | 0,00 € |
|  | Total |  | 381,12 € |
"""


def _verify(record):
    return ExtractorAgent._verify_string_spans(record, SOURCE)


def test_correct_numbers_verify_against_german_locale_tokens():
    record = {
        "items": [
            {"description": "Basic Fee wmView", "quantity": 1.0,
             "unit_price": 130.0, "total": 130.0},
            {"description": "Transaction Fee T1", "quantity": 14.0,
             "unit_price": 0.58, "total": 8.12},
        ],
        "grand_total": 381.12,
    }
    stats = _verify(record)
    assert not stats["flagged"], f"correct values must verify: {stats['flagged']}"
    assert not [r for r in stats["repaired"] if r["kind"] == "numeric_snap"]
    assert record["items"][1]["unit_price"] == 0.58


def test_digit_insertion_snaps_to_row_value():
    # All three live-observed transcription errors.
    record = {
        "items": [
            {"description": "Transaction Fee T3", "quantity": 162.0,
             "unit_price": 1.5, "total": 2243.0},           # source: 243,00
            {"description": "Basic Fee wmGuide", "quantity": 0.0,
             "unit_price": 1100.0, "total": 0.0},            # source: 1.000,00
            {"description": "Transaction Fee T6", "quantity": 0.0,
             "unit_price": 11.8, "total": 0.0},              # source: 1,80
        ],
    }
    stats = _verify(record)

    assert record["items"][0]["total"] == 243.0
    assert record["items"][1]["unit_price"] == 1000.0
    assert record["items"][2]["unit_price"] == 1.8
    snaps = [r for r in stats["repaired"] if r["kind"] == "numeric_snap"]
    assert len(snaps) == 3
    assert all(r["ratio"] >= 0.6 for r in snaps)
    assert not stats["flagged"]


def test_fabricated_number_is_flagged_not_snapped():
    record = {
        "items": [
            {"description": "Basic Fee wmView", "quantity": 1.0,
             "unit_price": 130.0, "total": 77777.77},  # nowhere near the row
        ],
    }
    stats = _verify(record)
    assert record["items"][0]["total"] == 77777.77, "no unsafe repair"
    numeric_flags = [f for f in stats["flagged"] if f.get("kind") == "numeric"]
    assert len(numeric_flags) == 1
    assert numeric_flags[0]["path"] == "items[0].total"


def test_doc_wide_verification_without_row_anchor():
    # grand_total has no string sibling row — verifies against the document.
    record = {"grand_total": 381.12}
    stats = _verify(record)
    assert stats["checked"] == 1 and stats["verified"] == 1
    assert not stats["flagged"]


def test_int_type_preserved_on_snap():
    record = {
        "items": [
            {"description": "Transaction Fee T3", "quantity": 1162,  # source: 162
             "unit_price": 1.5, "total": 243.0},
        ],
    }
    _verify(record)
    assert record["items"][0]["quantity"] == 162
    assert isinstance(record["items"][0]["quantity"], int)


def test_numeric_flags_do_not_trigger_string_retry():
    class _CountingClient:
        calls = 0

        def generate(self, *a, **k):
            _CountingClient.calls += 1
            return ""

        def generate_structured(self, *a, **k):
            _CountingClient.calls += 1
            return None

    from typing import List, Optional
    from pydantic import BaseModel

    class Item(BaseModel):
        quantity: Optional[float] = None

    class Rec(BaseModel):
        items: List[Item] = []

    agent = ExtractorAgent.__new__(ExtractorAgent)
    agent.client = _CountingClient()
    stats = {"flagged": [{"path": "items[0].quantity", "value": 9.9, "kind": "numeric"}]}

    audit = agent._retry_flagged_and_null_strings(
        record={"items": [{"quantity": 9.9}]},
        schema_model=Rec,
        source_markdown=SOURCE,
        grounding_stats=stats,
    )
    assert audit == []
    assert _CountingClient.calls == 0, "numeric flags must not consume retry LLM calls"
