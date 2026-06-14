"""Behavioural tests for calibrated confidence (task #52, audit killer #2).

`confidence_score` was a constant 1.0 — a default the model parroted and
nothing ever computed. The verifier now scores every assessed leaf by its
grounding outcome (verbatim/case-restored = 1.0, snapped = match ratio,
unverifiable = best similarity to any source span), and `_finalize_record`
overwrites every `confidence_score` key (record root and sub-blocks alike)
with the mean over the leaves under it. Snapped paths stay capped at their
snap ratio even after a post-retry refresh re-verifies them verbatim.
"""
from extractor.agent import ExtractorAgent

SOURCE = """<!-- page:1 -->
| Service Description | Amount | quantity | Total |
| --- | --- | --- | --- |
| Basic Fee wmView | 130,00 | 1 | 130,00 |
| Transaction Fee T3 | 1,50 | 162 | 243,00 |
TIBA PANAMA logistics services
"""


def _verify(record):
    return ExtractorAgent._verify_string_spans(record, SOURCE)


def test_all_verbatim_record_scores_one():
    record = {"description": "Basic Fee wmView", "total": 130.0,
              "company": "TIBA PANAMA logistics services"}
    stats = _verify(record)
    conf = stats["field_confidence"]
    assert set(conf.values()) == {1.0}
    assert len(conf) == 3

    record["confidence_score"] = 1.0  # the old parroted constant
    ExtractorAgent._apply_calibrated_confidence(record, conf)
    assert record["confidence_score"] == 1.0  # earned this time, not defaulted


def test_fuzzy_snap_scores_its_match_ratio():
    record = {"company": "TIBAANAMA logistics services"}  # char-drop typo
    stats = _verify(record)
    conf = stats["field_confidence"]
    assert 0.85 <= conf["company"] < 1.0
    snap = stats["repaired"][0]
    assert conf["company"] == snap["ratio"]


def test_fabrication_scores_low_and_drags_the_record_mean():
    record = {
        "description": "Basic Fee wmView",          # verbatim → 1.0
        "invented": "Quantum Flux Capacitor Module", # nowhere in source
        "confidence_score": 1.0,
    }
    stats = _verify(record)
    conf = stats["field_confidence"]
    assert conf["description"] == 1.0
    assert conf["invented"] < 0.7

    ExtractorAgent._apply_calibrated_confidence(record, conf)
    assert record["confidence_score"] < 1.0
    expected = round((conf["description"] + conf["invented"]) / 2, 4)
    assert record["confidence_score"] == expected


def test_numeric_outcomes_score_by_digit_ratio():
    record = {
        "items": [
            {"description": "Transaction Fee T3", "quantity": 162.0,
             "total": 2243.0},   # snaps to 243,00 in its row
            {"description": "Basic Fee wmView", "quantity": 1.0,
             "total": 77777.77}, # fabricated — flagged
        ],
    }
    stats = _verify(record)
    conf = stats["field_confidence"]
    snap = next(r for r in stats["repaired"] if r["kind"] == "numeric_snap")
    assert conf["items[0].total"] == snap["ratio"] < 1.0
    assert conf["items[1].total"] < 0.6, "flagged number scores its best digit ratio"
    assert conf["items[0].quantity"] == 1.0 and conf["items[1].quantity"] == 1.0


def test_sub_block_confidence_is_subtree_mean():
    record = {
        "header": {"company": "TIBA PANAMA logistics services",
                   "confidence_score": 1.0},
        "items": [{"description": "made up item xyz", "confidence_score": 1.0}],
        "confidence_score": 1.0,
    }
    stats = _verify(record)
    conf = stats["field_confidence"]
    ExtractorAgent._apply_calibrated_confidence(record, conf)

    assert record["header"]["confidence_score"] == 1.0
    assert record["items"][0]["confidence_score"] < 0.7
    expected_root = round(sum(conf.values()) / len(conf), 4)
    assert record["confidence_score"] == expected_root
    assert record["items"][0]["confidence_score"] < record["header"]["confidence_score"]


def test_unassessed_block_gets_none_not_certainty():
    record = {"empty_block": {"note": None, "confidence_score": 1.0},
              "company": "TIBA PANAMA logistics services",
              "confidence_score": 1.0}
    stats = _verify(record)
    ExtractorAgent._apply_calibrated_confidence(record, stats["field_confidence"])
    assert record["empty_block"]["confidence_score"] is None
    assert record["confidence_score"] == 1.0


def test_snap_ratio_survives_post_retry_refresh():
    # Pass 1: value snaps to the source span at ratio < 1.0.
    record = {"company": "TIBAANAMA logistics services"}
    stats1 = _verify(record)
    snap_ratio = stats1["repaired"][0]["ratio"]
    assert snap_ratio < 1.0

    # Refresh (as _finalize_record does): the repaired value now verifies
    # verbatim, but the carried-over repair audit must cap its confidence.
    stats2 = _verify(record)
    assert stats2["field_confidence"]["company"] == 1.0  # verbatim now
    stats2["repaired"] = stats1["repaired"] + stats2["repaired"]

    clamped = ExtractorAgent._calibrated_field_confidence(stats2)
    assert clamped["company"] == snap_ratio


def test_schema_defaults_are_no_longer_constant_one():
    from extractor.schema_definitions import LibrarianUniversalHardware
    rec = LibrarianUniversalHardware()
    assert rec.confidence_score is None, "default must mean 'not assessed', not 1.0"


def test_source_evidence_metadata_is_not_audited():
    # source_evidence is provenance the pipeline itself attaches; auditing
    # its snippets/page numbers as extracted content inflated `checked` and
    # flagged snippets the verifier produced in the first place.
    record = {
        "company": "TIBA PANAMA logistics services",
        "source_evidence": [
            {"text_snippet": "totally absent snippet xyz",
             "page_number": 99, "confidence": 0.5},
        ],
        "items": [
            {"description": "Basic Fee wmView",
             "source_evidence": {"text_snippet": "also absent qqq",
                                 "page_number": 42, "confidence": 0.9}},
        ],
    }
    stats = _verify(record)
    assert stats["checked"] == 2, "only company + items[0].description"
    assert not stats["flagged"]
    assert all("source_evidence" not in p for p in stats["field_confidence"])
    assert record["source_evidence"][0]["text_snippet"] == "totally absent snippet xyz"
