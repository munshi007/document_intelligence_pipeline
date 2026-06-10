"""Behavioural tests for list-abbreviation handling in the JSON salvage path
(task #48, fix B).

The extraction model sometimes abbreviates a long array with a bare `...`
element ("and so on"). Before this fix the json_repair fallback absorbed every
key AFTER the abbreviated list (grand_total, confidence_score, ...) into a
zombie list item, which was then dropped — correct data the model actually
produced was silently lost. These tests lock in:

  1. Detection (`_has_list_abbreviation`) fires on structural ellipses only,
     never on ellipses inside string values.
  2. Salvage (`_extract_json_payload`) preserves the keys following an
     abbreviated list and yields only complete list items.

The fixture mirrors the REAL failure artifact from sample-invoice
(debug trace batch_0001_attempt1_raw.txt, 2026-06-10).
"""
import json

from common.vlm_providers.local_text_provider import LocalTextProvider

# Condensed copy of the real failure shape: thought preamble, fenced JSON,
# items abbreviated after the 2nd entry with a dangling `{` + `...`, then a
# premature `}` before the remaining root keys.
ABBREVIATED_OUTPUT = """<thought>
Reasoning about the invoice.
</thought>

```json
{
  "identity": {"number": "123100401"},
  "items": [
    {
      "description": "basic fee wmview",
      "quantity": 1.0,
      "total": 130.0
    },
    {
      "description": "transaction fee t1",
      "quantity": 14.0,
      "total": 8.12
    },
    {
    ...
    ]
},
"grand_total": 381.12,
"page_references": [1],
"confidence_score": 1.0
}
```"""


def test_keys_after_abbreviated_list_survive_salvage():
    payload = LocalTextProvider._extract_json_payload(
        ABBREVIATED_OUTPUT, expected_type="object"
    )
    assert payload, "salvage must recover the malformed JSON"
    obj = json.loads(payload)

    assert obj["grand_total"] == 381.12, "key after abbreviated list was lost"
    assert obj["confidence_score"] == 1.0
    assert obj["identity"]["number"] == "123100401"
    assert len(obj["items"]) == 2, "only complete items, no zombie from the dangling '{'"
    assert all(it.get("description") for it in obj["items"])


def test_detection_fires_on_structural_ellipsis():
    assert LocalTextProvider._has_list_abbreviation(ABBREVIATED_OUTPUT) is True
    assert LocalTextProvider._has_list_abbreviation('{"a": [1, ..., 9]}') is True
    assert LocalTextProvider._has_list_abbreviation('{"a": [..., {"b": 1}]}') is True
    assert LocalTextProvider._has_list_abbreviation('{"a": [{"b": 1}, …]}') is True


def test_detection_ignores_ellipsis_inside_strings():
    assert LocalTextProvider._has_list_abbreviation(
        '{"note": "to be continued..."}'
    ) is False
    assert LocalTextProvider._has_list_abbreviation(
        '{"items": [{"description": "wait for it ..."}]}'
    ) is False
    assert LocalTextProvider._has_list_abbreviation('{"a": [1, 2, 3]}') is False
    assert LocalTextProvider._has_list_abbreviation("") is False


def test_strip_leaves_string_ellipses_untouched():
    src = '{"note": "to be continued...", "items": [1, ..., 9]}'
    out = LocalTextProvider._strip_list_abbreviations(src)
    assert '"to be continued..."' in out, "string content must not be mutated"

    # Full salvage ladder handles the comma debris the strip leaves behind.
    payload = LocalTextProvider._extract_json_payload(src, expected_type="object")
    obj = json.loads(payload)
    assert obj["note"] == "to be continued..."
    assert obj["items"] == [1, 9]


def test_valid_json_passthrough_unchanged():
    clean = '{"items": [{"description": "a"}, {"description": "b"}], "grand_total": 5.0}'
    payload = LocalTextProvider._extract_json_payload(
        f"```json\n{clean}\n```", expected_type="object"
    )
    assert json.loads(payload) == json.loads(clean)


COMPLETE_OUTPUT = """```json
{
  "identity": {"number": "123100401"},
  "items": [
    {"description": "basic fee wmview", "quantity": 1.0, "total": 130.0},
    {"description": "transaction fee t1", "quantity": 14.0, "total": 8.12},
    {"description": "transaction fee t3", "quantity": 162.0, "total": 243.0}
  ],
  "grand_total": 381.12
}
```"""


def test_abbreviated_attempt_triggers_regeneration_and_uses_complete_output():
    """Fix A: when attempt 1 abbreviates, generate_structured regenerates with
    the anti-abbreviation instruction and prefers the complete output."""
    from typing import List, Optional
    from pydantic import BaseModel

    class Identity(BaseModel):
        number: Optional[str] = None

    class Item(BaseModel):
        description: Optional[str] = None
        quantity: Optional[float] = None
        total: Optional[float] = None

    class Record(BaseModel):
        identity: Optional[Identity] = None
        items: List[Item] = []
        grand_total: Optional[float] = None

    provider = LocalTextProvider.__new__(LocalTextProvider)  # no model load
    calls = []

    def fake_run_generation(full_prompt, max_tokens):
        calls.append(full_prompt)
        return ABBREVIATED_OUTPUT if len(calls) == 1 else COMPLETE_OUTPUT

    provider._run_generation = fake_run_generation

    prev_model = LocalTextProvider._model
    LocalTextProvider._model = object()  # pass the loaded-model guard
    try:
        result = provider.generate_structured(
            image=None, prompt="extract", response_model=Record
        )
    finally:
        LocalTextProvider._model = prev_model

    assert len(calls) == 2, "abbreviated attempt must trigger one regeneration"
    assert "NEVER" in calls[1] and "ellipsis" in calls[1]
    assert result is not None
    assert len(result.items) == 3, "complete regenerated output must win"
    assert result.grand_total == 381.12


def test_grounding_summary_pass_rate_capped_with_case_restores():
    """Stats fix: case_restore repairs are already inside `verified` — the
    stage summary must not add them to grounded again (pass_rate > 1.0)."""
    from stages.extract import _grounding_summary

    # 6 fields checked: 5 verified (2 of them via case restore), 1 fuzzy
    # snapped (fuzzy repairs are NOT counted in verified) → all 6 grounded.
    stats = {
        "checked": 6,
        "verified": 5,
        "repaired": [
            {"path": "from.name", "ratio": 1.0, "kind": "case_restore"},
            {"path": "to.name", "ratio": 1.0, "kind": "case_restore"},
            {"path": "company", "ratio": 0.93, "kind": "fuzzy_snap"},
        ],
        "flagged": [],
    }
    summary = _grounding_summary(stats)
    assert summary["pass_rate"] == 1.0
    assert summary["repaired"] == 3

    realistic = {
        "checked": 12,
        "verified": 12,
        "repaired": [
            {"path": f"items[{i}].description", "ratio": 1.0, "kind": "case_restore"}
            for i in range(10)
        ],
        "flagged": [],
    }
    summary = _grounding_summary(realistic)
    assert summary["pass_rate"] == 1.0, "10 case restores must not inflate pass_rate"
