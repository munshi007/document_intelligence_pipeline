"""Behavioural tests for source-case restoration in the grounding verifier
(task #47).

The verifier matches extracted values against the source case-insensitively,
so a case-mangled model output ("CPb software (germany) gmbh") used to be
certified verbatim ratio 1.0 and never repaired. These tests lock in the new
invariant: the source is ground truth for casing — any value that matches the
source only case-insensitively is snapped back to the source's own rendering,
and the snap is recorded in stats["repaired"] so pass_rate stays auditable.

Exercises the REAL staticmethods (`_verify_string_spans`,
`_verify_retry_response`) — no agent instance or model load required.
"""
from extractor.agent import ExtractorAgent

SOURCE = """<!-- page:1 -->

CPB Software (Germany) GmbH - Im Bruch 3 - 63897 Miltenberg/Main
<!-- id:p1_r2 page:1 type:text bbox:[58.1,160.1,275.8,169.2] src:layout_model conf:0.70 -->

Musterkunde AG
Mr. John Doe
Musterstr. 23
12345 Musterstadt

| Invoice No | Date |
| --- | --- |
| 123100401 | 1. März |

TIBA PANAMA logistics services
"""


def test_case_mangled_verbatim_value_is_restored_to_source_case():
    record = {"from": {"name": "CPb software (germany) gmbh"}}

    stats = ExtractorAgent._verify_string_spans(record, SOURCE)

    assert record["from"]["name"] == "CPB Software (Germany) GmbH"
    assert stats["checked"] == 1 and stats["verified"] == 1
    assert len(stats["repaired"]) == 1
    rep = stats["repaired"][0]
    assert rep["before"] == "CPb software (germany) gmbh"
    assert rep["after"] == "CPB Software (Germany) GmbH"
    assert rep["ratio"] == 1.0
    assert not stats["flagged"]


def test_correctly_cased_value_is_untouched():
    record = {"name": "CPB Software (Germany) GmbH"}

    stats = ExtractorAgent._verify_string_spans(record, SOURCE)

    assert record["name"] == "CPB Software (Germany) GmbH"
    assert stats["verified"] == 1
    assert not stats["repaired"], "no-op values must not produce repair noise"


def test_multiline_value_keeps_source_newlines():
    record = {"to": {"name": "musterkunde ag\nmr. john doe"}}

    stats = ExtractorAgent._verify_string_spans(record, SOURCE)

    assert record["to"]["name"] == "Musterkunde AG\nMr. John Doe"
    assert stats["verified"] == 1 and len(stats["repaired"]) == 1


def test_fuzzy_repair_path_still_works():
    # Character-drop mangling (not just case) must still go through the
    # fuzzy snap path — the original purpose of the verifier.
    record = {"company": "TIBAANAMA logistics services"}

    stats = ExtractorAgent._verify_string_spans(record, SOURCE)

    assert record["company"] == "TIBA PANAMA logistics services"
    assert len(stats["repaired"]) == 1
    assert stats["repaired"][0]["ratio"] < 1.0, "fuzzy snap, not verbatim"


def test_unfindable_value_is_flagged_not_repaired():
    record = {"name": "completely fabricated value xyz"}

    stats = ExtractorAgent._verify_string_spans(record, SOURCE)

    assert record["name"] == "completely fabricated value xyz"
    assert stats["verified"] == 0
    assert len(stats["flagged"]) == 1


def test_retry_verbatim_returns_source_cased_span():
    accepted, value, detail = ExtractorAgent._verify_retry_response(
        "cpb software (germany) gmbh", SOURCE
    )

    assert accepted is True
    assert value == "CPB Software (Germany) GmbH"
    assert detail == "verbatim_match"


def test_retry_exact_response_passes_through():
    accepted, value, detail = ExtractorAgent._verify_retry_response(
        "Musterstr. 23", SOURCE
    )

    assert accepted is True
    assert value == "Musterstr. 23"
    assert detail == "verbatim_match"


def test_retry_not_in_source_still_rejected():
    accepted, value, detail = ExtractorAgent._verify_retry_response(
        "completely fabricated value xyz", SOURCE
    )

    assert accepted is False and value is None
