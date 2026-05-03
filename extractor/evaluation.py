"""
Extraction Evaluation Utilities.

Computes per-run quality scorecards for schema-conditioned extraction.

Scorecard metrics
-----------------
schema_validity_rate        — 1.0 if all required fields present, else 0.0
required_field_completion_rate — fraction of required fields that are non-empty
non_empty_extraction_rate   — 1.0 if at least one field populated
grounded_reference_rate     — 1.0 if page_references present
retry_frequency             — parse_failures / total_batches

Ground-truth metrics (populated only when ground_truth dict is passed)
gt_field_precision          — precision against annotated gold fields
gt_field_recall             — recall against annotated gold fields
gt_field_f1                 — harmonic mean of precision + recall
matched_fields              — list of fields correctly matched
missing_fields              — list of GT fields not found in extraction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _populated_field_count(payload: Dict[str, Any], ignore_fields: List[str] | None = None) -> int:
    ignore = set(ignore_fields or [])
    count = 0
    for key, value in payload.items():
        if key in ignore:
            continue
        if not _is_empty_value(value):
            count += 1
    return count


def _required_field_stats(payload: Dict[str, Any], schema_json: Dict[str, Any]) -> Dict[str, Any]:
    required = schema_json.get("required", []) if isinstance(schema_json, dict) else []
    if not required:
        return {
            "required_total": 0,
            "required_missing": [],
            "required_field_completion_rate": 1.0,
        }

    missing = []
    for field in required:
        if _is_empty_value(payload.get(field)):
            missing.append(field)

    completion = (len(required) - len(missing)) / len(required)
    return {
        "required_total": len(required),
        "required_missing": missing,
        "required_field_completion_rate": round(completion, 4),
    }


def _compute_gt_metrics(
    extraction: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare extraction output against a ground-truth annotation dict.

    Matching is lenient: a field is 'correct' when the GT value is a
    case-insensitive substring of the extracted value (or vice versa).
    This handles minor formatting differences without requiring exact match.

    Handles FieldValue envelopes (dicts with raw_value/normalized_value)
    transparently.
    """
    if not ground_truth:
        return {}

    gt_fields = set(ground_truth.keys())
    ext_fields = set(k for k, v in extraction.items() if not _is_empty_value(v))

    matched: List[str] = []
    for field in gt_fields:
        gt_val = str(ground_truth.get(field, "")).lower().strip()
        raw_ext = extraction.get(field)
        # Unwrap FieldValue envelope if present
        if isinstance(raw_ext, dict) and ("normalized_value" in raw_ext or "raw_value" in raw_ext):
            ext_val = str(
                raw_ext.get("normalized_value") or raw_ext.get("raw_value") or ""
            ).lower().strip()
        else:
            ext_val = str(raw_ext or "").lower().strip()

        if gt_val and ext_val and (gt_val in ext_val or ext_val in gt_val):
            matched.append(field)

    n_matched = len(matched)
    precision = n_matched / len(ext_fields) if ext_fields else 0.0
    recall = n_matched / len(gt_fields) if gt_fields else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "gt_field_precision": round(precision, 4),
        "gt_field_recall": round(recall, 4),
        "gt_field_f1": round(f1, 4),
        "matched_fields": matched,
        "missing_fields": sorted(gt_fields - set(matched)),
        "extra_fields": sorted(ext_fields - gt_fields),
    }


def _parse_failure_entries(parse_failure_file: Path) -> List[Dict[str, Any]]:
    if not parse_failure_file.exists():
        return []

    rows: List[Dict[str, Any]] = []
    for line in parse_failure_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def evaluate_extraction(
    extraction_payload: Dict[str, Any],
    target_schema_json: Dict[str, Any],
    doc_stem: str,
    trace_dir: str | None = None,
    ground_truth: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute a single-run extraction scorecard."""

    failed = extraction_payload.get("status") == "failed"

    if failed:
        body = extraction_payload.get("details", {})
        data_payload: Dict[str, Any] = {}
        schema_title = body.get("schema") or target_schema_json.get("title", "unknown")
    else:
        body = extraction_payload
        data_payload = extraction_payload
        schema_title = target_schema_json.get("title", body.get("schema_title", "unknown"))

    required_stats = _required_field_stats(data_payload, target_schema_json)

    # 1. Schema validity rate (single-doc run => 0 or 1)
    schema_validity_rate = 0.0 if failed else (1.0 if not required_stats["required_missing"] else 0.0)

    # 2. Required-field completion rate
    required_completion_rate = required_stats["required_field_completion_rate"]

    # 3. Non-empty extraction rate (single-doc run => 0 or 1)
    populated = _populated_field_count(
        data_payload,
        ignore_fields=["confidence_score", "reasoning_thoughts", "page_references", "status", "error", "details"],
    )
    non_empty_extraction_rate = 1.0 if (not failed and populated > 0) else 0.0

    # 4. Grounded reference rate (single-doc run => 0 or 1)
    page_refs = data_payload.get("page_references", []) if isinstance(data_payload, dict) else []
    grounded_reference_rate = 1.0 if isinstance(page_refs, list) and len(page_refs) > 0 else 0.0

    # 5. Retry frequency
    retry_frequency = 0.0
    parse_failure_count = 0
    if trace_dir:
        pf = Path(trace_dir) / f"{doc_stem}_parse_failures.jsonl"
        entries = _parse_failure_entries(pf)
        parse_failure_count = len(entries)
        total_batches = body.get("total_batches") or 1
        retry_frequency = round(parse_failure_count / max(int(total_batches), 1), 4)

    return {
        "doc": doc_stem,
        "schema_title": schema_title,
        "status": "failed" if failed else "success",
        "schema_validity_rate": round(schema_validity_rate, 4),
        "required_field_completion_rate": round(required_completion_rate, 4),
        "non_empty_extraction_rate": round(non_empty_extraction_rate, 4),
        "grounded_reference_rate": round(grounded_reference_rate, 4),
        "retry_frequency": retry_frequency,
        "required_total": required_stats["required_total"],
        "required_missing": required_stats["required_missing"],
        "parse_failure_count": parse_failure_count,
        **(_compute_gt_metrics(data_payload, ground_truth) if ground_truth and not failed else {}),
    }
