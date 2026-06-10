"""Behavioural tests for the recall coverage check + targeted list recovery
(task #50).

Every prior check is a precision check — it judges values that are present.
A model that returns items:[] with a rationalization scores populated=1.0 and
grounding pass_rate 1.0. The coverage pass closes that blind spot:

  1. Detect source tables whose data rows mostly never made it into the
     extracted record.
  2. Re-extract EMPTY list fields from the matched table (header↔field-name
     affinity), accepting only items that ground in the table text.
  3. Surface everything in grounding stats — no more silent recall loss.
"""
from typing import List, Optional

from pydantic import BaseModel

from extractor.agent import ExtractorAgent


SOURCE = """<!-- page:1 -->
Invoice WMACCESS Internet

| Invoice No | Date |
| --- | --- |
| 123100401 | 1. März |

| Service Description | Amount -without VAT- | quantity | Total Amount |
| --- | --- | --- | --- |
| Basic Fee wmView | 130,00 € | 1 | 130,00 € |
| Transaction Fee T1 | 0,58 € | 14 | 8,12 € |
| Transaction Fee T3 | 1,50 € | 162 | 243,00 € |
| Transaction Fee T4 | 0,50 € | 0 | 0,00 € |
| Transaction Fee T5 | 0,80 € | 0 | 0,00 € |

Terms of Payment: Immediate payment without discount.
"""


class Item(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    total: Optional[float] = None


class Record(BaseModel):
    items: List[Item] = []
    grand_total: Optional[float] = None


def _agent(client=None) -> ExtractorAgent:
    agent = ExtractorAgent.__new__(ExtractorAgent)  # bypass __init__
    agent.client = client
    return agent


def test_markdown_tables_parse_headers_and_rows():
    tables = ExtractorAgent._markdown_tables(SOURCE)
    assert len(tables) == 2
    assert tables[0]["header"] == ["Invoice No", "Date"]
    assert len(tables[0]["rows"]) == 1
    assert tables[1]["header"][0] == "Service Description"
    assert len(tables[1]["rows"]) == 5
    assert tables[1]["rows"][2][0] == "Transaction Fee T3"


def test_empty_items_flags_the_table():
    record = {"items": [], "grand_total": None}
    flags, tables = _agent()._undercovered_tables(record, SOURCE)
    assert len(flags) == 1, "the 5-row service table must be flagged"
    assert flags[0]["data_rows"] == 5
    assert flags[0]["covered_rows"] == 0
    # the 1-row header table is below min_rows and must NOT be flagged
    assert all(f["table_index"] != 0 for f in flags)


def test_full_extraction_is_not_flagged():
    record = {
        "items": [
            {"description": "Basic Fee wmView", "total": 130.0},
            {"description": "Transaction Fee T1", "total": 8.12},
            {"description": "Transaction Fee T3", "total": 243.0},
            {"description": "Transaction Fee T4", "total": 0.0},
            {"description": "Transaction Fee T5", "total": 0.0},
        ],
    }
    flags, _ = _agent()._undercovered_tables(record, SOURCE)
    assert flags == []


def test_header_field_affinity_is_schema_driven():
    aff = ExtractorAgent._header_field_affinity(
        ["Service Description", "Amount -without VAT-", "quantity", "Total Amount"],
        Item,
    )
    assert aff == 1.0  # description, quantity, total all appear in the header

    numeric_header_aff = ExtractorAgent._header_field_affinity(
        ["4", "0", "9", "15,82 €"], Item
    )
    assert numeric_header_aff == 0.0, "junk numeric tables must score zero"


class _RecoveryClient:
    """Stub: returns a recovered list with one fabricated item that must be
    dropped by the table-grounding gate."""

    def __init__(self):
        self.prompts = []

    def generate_structured(self, image, prompt, response_model, **kwargs):
        self.prompts.append(prompt)
        return response_model(items=[
            Item(description="Basic Fee wmView", quantity=1.0, total=130.0),
            Item(description="Transaction Fee T1", quantity=14.0, total=8.12),
            Item(description="Transaction Fee T3", quantity=162.0, total=243.0),
            Item(description="Premium Support Plan", quantity=1.0, total=999.0),  # fabricated
        ])


def test_recovery_fills_empty_list_and_drops_fabricated_items():
    client = _RecoveryClient()
    agent = _agent(client)
    record = {"items": [], "grand_total": None}

    coverage = agent._audit_list_coverage(record, Record, SOURCE)

    assert coverage is not None
    assert coverage["tables_checked"] == 1
    assert len(coverage["recovered"]) == 1
    rec = coverage["recovered"][0]
    assert rec["field"] == "items" and rec["accepted"] == 3
    assert len(record["items"]) == 3, "empty list filled with grounded items only"
    descs = [i["description"] for i in record["items"]]
    assert "Premium Support Plan" not in descs, "fabricated item must be dropped"
    assert coverage["undercovered"] == [], "table is covered after recovery"
    # the focused prompt must carry the actual table text
    assert "Transaction Fee T3" in client.prompts[0]


def test_recovery_never_mutates_nonempty_lists():
    client = _RecoveryClient()
    agent = _agent(client)
    existing = [{"description": "Basic Fee wmView", "quantity": 1.0, "total": 130.0}]
    record = {"items": list(existing), "grand_total": None}

    coverage = agent._audit_list_coverage(record, Record, SOURCE)

    assert record["items"] == existing, "non-empty lists are flag-only"
    assert client.prompts == [], "no LLM call when no empty list field exists"
    assert len(coverage["undercovered"]) == 1, "deficit stays visible as a flag"


def test_no_tables_returns_none():
    coverage = _agent()._audit_list_coverage(
        {"items": []}, Record, "Just prose, no tables here."
    )
    assert coverage is None
