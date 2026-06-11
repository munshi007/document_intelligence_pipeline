"""Behavioural tests for block-level SourceEvidence grounding (task #46).

The old `_ground_block` was a stub: it fired an LLM call per block whose
result was discarded (and errored — langextract rejects `examples=[]`), then
overwrote `block.source_evidence` with a fabricated `context[:200]` snippet
at `page_number=1, confidence=0.95`. A hardcoded block whitelist
(electrical/mechanical/connectors/diagnostics/invoice_header — killer-#3)
limited the blast radius. Both are gone: any sub-model declaring a
`source_evidence` field is grounded deterministically via the span verifier,
with the real snippet, real page (from <!-- page:N --> markers) and the
match ratio as confidence. No LLM call, no whitelist, no fabrication.
"""
from typing import List, Optional

from pydantic import BaseModel

from extractor.agent import ExtractorAgent
from extractor.schema_definitions import SourceEvidence

SOURCE = """<!-- page:1 -->
Invoice No: 123100401
Date: 1. März 2024
<!-- page:2 -->
Warranty period: 24 months from delivery
"""


class Header(BaseModel):
    invoice_no: Optional[str] = None
    source_evidence: Optional[SourceEvidence] = None


class Warranty(BaseModel):
    # Deliberately NOT in the old whitelist — proves schema-agnosticism.
    terms: Optional[str] = None
    source_evidence: Optional[SourceEvidence] = None


class LineItem(BaseModel):
    description: Optional[str] = None
    source_evidence: Optional[SourceEvidence] = None


class Record(BaseModel):
    invoice_header: Optional[Header] = None
    warranty: Optional[Warranty] = None
    items: List[LineItem] = []


def _agent() -> ExtractorAgent:
    return ExtractorAgent.__new__(ExtractorAgent)  # no model/client load


def test_block_gets_real_evidence_with_page_and_ratio():
    master = Record(invoice_header=Header(invoice_no="123100401"))
    _agent()._ground_block_evidence(master, SOURCE)

    ev = master.invoice_header.source_evidence
    assert ev is not None
    assert ev.text_snippet == "123100401", "snippet must be the verified span, not context[:200]"
    assert ev.page_number == 1
    assert ev.confidence == 1.0


def test_non_whitelisted_block_is_grounded_too():
    master = Record(warranty=Warranty(terms="24 months from delivery"))
    _agent()._ground_block_evidence(master, SOURCE)

    ev = master.warranty.source_evidence
    assert ev is not None, "blocks outside the old whitelist must be grounded"
    assert ev.page_number == 2


def test_list_item_blocks_are_grounded():
    master = Record(items=[LineItem(description="Warranty period"),
                           LineItem(description="not in the document at all xyz")])
    _agent()._ground_block_evidence(master, SOURCE)

    assert master.items[0].source_evidence is not None
    assert master.items[0].source_evidence.page_number == 2
    assert master.items[1].source_evidence is None, "no fabricated evidence for ungrounded values"


def test_model_produced_evidence_is_never_overwritten():
    own = SourceEvidence(text_snippet="model's own citation", page_number=2, confidence=0.8)
    master = Record(invoice_header=Header(invoice_no="123100401", source_evidence=own))
    _agent()._ground_block_evidence(master, SOURCE)

    assert master.invoice_header.source_evidence is own


def test_ungroundable_block_left_blank():
    master = Record(invoice_header=Header(invoice_no="999999999"))  # not in source
    _agent()._ground_block_evidence(master, SOURCE)
    assert master.invoice_header.source_evidence is None


def test_list_annotated_evidence_field_gets_a_list():
    class ListEvidenceBlock(BaseModel):
        name: Optional[str] = None
        source_evidence: List[SourceEvidence] = []

    class Rec(BaseModel):
        block: Optional[ListEvidenceBlock] = None

    master = Rec(block=ListEvidenceBlock(name="123100401"))
    _agent()._ground_block_evidence(master, SOURCE)

    ev = master.block.source_evidence
    assert isinstance(ev, list) and len(ev) == 1
    assert ev[0].text_snippet == "123100401"


def test_no_llm_call_is_made():
    class _ExplodingClient:
        def __getattr__(self, name):
            raise AssertionError("block grounding must not touch the LLM client")

    agent = _agent()
    agent.client = _ExplodingClient()
    master = Record(invoice_header=Header(invoice_no="123100401"),
                    warranty=Warranty(terms="24 months from delivery"))
    agent._ground_block_evidence(master, SOURCE)  # must not raise
    assert master.invoice_header.source_evidence is not None
