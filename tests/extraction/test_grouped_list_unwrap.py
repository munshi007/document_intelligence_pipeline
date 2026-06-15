"""Recall-stability tests for group-nested list unwrapping (task #53).

The extraction model intermittently emits a *grouped* shape for a
``List[ItemModel]`` field — wrapping the real rows in an off-schema
sub-list — instead of the flat list the schema declares::

    "parameters": [{"name": "Operating Voltage DC",
                    "values": [ {param, min_value, max_value, unit, value}, ... ]}]

Pydantic keeps only the wrapper and silently drops the inner array, which
collapsed a 34-parameter M12 datasheet down to 1. ``_unwrap_grouped_list_items``
flattens the inner rows up (schema-driven, keyed only on the item model's own
field names), ``_realign_leftover_keys`` maps the drifted ``param`` -> ``name``
and synthesizes ``value`` from a ``min/max`` range, and ``_tolerant_list_salvage``
keeps a record by dropping only the individual rows that fail validation.
"""
from common.vlm_providers.local_text_provider import LocalTextProvider as P
from extractor.schema_definitions import (
    LibrarianUniversalHardware as Hardware,
    TechParameter,
    ConnectorSpec,
)

# The exact wrapper shape captured in the #53 debug trace (abridged).
GROUPED_DATASHEET = {
    "art_no": "7000-12231-2140200",
    "parameters": [
        {
            "name": "Operating Voltage DC",
            "values": [
                {"param": "DC min.", "min_value": "18", "max_value": None, "unit": "V", "value": None},
                {"param": "DC max.", "min_value": None, "max_value": "30", "unit": "V", "value": None},
                {"param": "Operating Voltage AC", "min_value": None, "max_value": None, "unit": "V", "value": "300"},
                {"param": "Material Group", "min_value": None, "max_value": None, "unit": None, "value": "I"},
                {"param": "Coating of Contact", "min_value": None, "max_value": None, "unit": None, "value": "gold plated"},
            ],
        }
    ],
    "connectors": [],
}


def test_grouped_parameters_flatten_to_real_rows():
    out = P._unwrap_grouped_list_items(dict(GROUPED_DATASHEET), Hardware)
    params = out["parameters"]
    assert len(params) == 5, "all 5 inner rows recovered, not the 1 wrapper"
    names = [p["name"] for p in params]
    assert "DC min." in names and "Operating Voltage AC" in names
    assert "Operating Voltage DC" not in names, "wrapper title is not a row"


def test_flattened_record_validates_with_all_rows():
    out = P._unwrap_grouped_list_items(dict(GROUPED_DATASHEET), Hardware)
    rec = Hardware.model_validate(out)
    assert len(rec.parameters) == 5
    by_name = {p.name: p for p in rec.parameters}
    # param -> name realignment
    assert by_name["Operating Voltage AC"].value == "300"
    # value synthesized from a min/max-only row (value was null)
    assert by_name["DC min."].value == "18"
    assert by_name["DC max."].value == "30"


def test_already_flat_list_is_untouched():
    flat = {"parameters": [
        {"name": "Voltage", "value": "24", "unit": "V"},
        {"name": "Current", "value": "4", "unit": "A"},
    ]}
    out = P._unwrap_grouped_list_items(dict(flat), Hardware)
    assert out["parameters"] == flat["parameters"], "no false unwrap on flat rows"
    assert len(Hardware.model_validate(out).parameters) == 2


def test_declared_nested_field_is_not_unwrapped():
    # ConnectorSpec.pins is a *declared* nested list — it must never be
    # mistaken for a group wrapper and flattened into the connectors list.
    rec = {"connectors": [
        {"name": "Port 0", "type": "M12", "pins": [
            {"pin": "1", "signal": "V+"}, {"pin": "2", "signal": "Bus A"},
        ]},
    ]}
    out = P._unwrap_grouped_list_items(dict(rec), Hardware)
    assert len(out["connectors"]) == 1, "connector wrapper preserved"
    assert out["connectors"][0]["name"] == "Port 0"
    assert len(out["connectors"][0]["pins"]) == 2


def test_realign_maps_drifted_key_to_missing_required():
    item = {"param": "Stripping Length", "value": "22", "unit": "mm"}
    P._realign_leftover_keys(item, set(TechParameter.model_fields), ["name", "value"])
    assert item["name"] == "Stripping Length"
    assert "param" not in item


def test_realign_synthesizes_value_from_range():
    item = {"name": "Temp", "min_value": "-25", "max_value": "85", "unit": "C"}
    P._realign_leftover_keys(item, set(TechParameter.model_fields), ["name", "value"])
    assert item["value"] == "-25...85"


def test_tolerant_salvage_drops_only_bad_rows():
    payload = {"parameters": [
        {"name": "Good A", "value": "1"},
        {"name": "Bad", "value": None},          # value is required -> invalid
        {"name": "Good B", "value": "2"},
    ]}
    rec = P._tolerant_list_salvage(payload, Hardware)
    assert rec is not None
    kept = [p.name for p in rec.parameters]
    assert kept == ["Good A", "Good B"], "bad row dropped, good rows kept"


def test_tolerant_salvage_passthrough_when_already_valid():
    payload = {"parameters": [{"name": "V", "value": "24", "unit": "V"}]}
    rec = P._tolerant_list_salvage(payload, Hardware)
    assert rec is not None and len(rec.parameters) == 1


def test_scalar_list_field_is_ignored():
    # page_references is List[int] — _list_item_model must return None so the
    # unwrap pass skips it entirely (no BaseModel item type).
    assert P._list_item_model(Hardware.model_fields["page_references"].annotation) is None
    rec = {"page_references": [1, 2, 3], "parameters": []}
    out = P._unwrap_grouped_list_items(dict(rec), Hardware)
    assert out["page_references"] == [1, 2, 3]
