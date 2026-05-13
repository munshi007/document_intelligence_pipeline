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


def _get_nested(obj: Any, path: str) -> Any:
    """
    Resolve a dotted path against a possibly-nested dict/list.
    Numeric segments index into lists, e.g. 'charges.0.value'.
    Returns None if any segment misses.
    """
    if obj is None or not path:
        return None
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError, TypeError):
                return None
        else:
            return None
    return cur


def _matches_gt(gt_val: Any, ext_val_raw: Any) -> bool:
    """
    Lenient substring match. Three GT-value flavours are supported:

      • str / number / bool : case-insensitive substring (either direction).
      • list[str]           : every expected substring must appear somewhere
                              in the stringified extraction at that path
                              (good for asserting "these line items exist").
      • dict                : same as str — stringify and substring-match.

    FieldValue envelopes ({raw_value, normalized_value, ...}) are unwrapped.
    """
    if isinstance(ext_val_raw, dict) and (
        "normalized_value" in ext_val_raw or "raw_value" in ext_val_raw
    ):
        ext_str = str(
            ext_val_raw.get("normalized_value") or ext_val_raw.get("raw_value") or ""
        ).lower().strip()
    else:
        ext_str = str(ext_val_raw or "").lower().strip()

    if not ext_str:
        return False

    if isinstance(gt_val, list):
        if not gt_val:
            return False
        return all(
            str(item).lower().strip() in ext_str
            for item in gt_val if str(item).strip()
        )

    gt_str = str(gt_val).lower().strip()
    if not gt_str:
        return False
    return gt_str in ext_str or ext_str in gt_str


def _compute_gt_metrics(
    extraction: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare extraction output against a ground-truth annotation dict.

    Accepted GT shapes:
      • Flat field map:               {"quote_no": "1922895", ...}
      • Wrapped per project README:   {"doc_id": ..., "fields": {...}}

    GT keys may be dotted paths into the extraction tree, e.g.
    'quote_identity.quote_no' or 'charges.0.description'. List-valued GT
    entries assert that every expected substring shows up at that path.

    Precision denominator: count of *top-level* populated extraction fields
    (a coarse but stable signal); recall denominator: count of GT entries.
    """
    if not ground_truth:
        return {}

    # Unwrap project-convention envelope.
    if isinstance(ground_truth.get("fields"), dict):
        ground_truth = ground_truth["fields"]

    gt_fields = set(ground_truth.keys())
    ext_fields = set(k for k, v in extraction.items() if not _is_empty_value(v))

    matched: List[str] = []
    for field in gt_fields:
        gt_val = ground_truth.get(field)
        raw_ext = _get_nested(extraction, field)
        if _matches_gt(gt_val, raw_ext):
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
        "matched_fields": sorted(matched),
        "missing_fields": sorted(gt_fields - set(matched)),
        "extra_fields": sorted(ext_fields - {f.split(".")[0] for f in gt_fields}),
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
    grounding_stats: Optional[Dict[str, Any]] = None,
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
    # Satisfied by either page_references[] (any page cited) or
    # source_evidence[] (any text snippet supplied with provenance).
    page_refs = data_payload.get("page_references", []) if isinstance(data_payload, dict) else []
    evidence = data_payload.get("source_evidence", []) if isinstance(data_payload, dict) else []
    has_refs = isinstance(page_refs, list) and len(page_refs) > 0
    has_evidence = isinstance(evidence, list) and len(evidence) > 0
    grounded_reference_rate = 1.0 if (has_refs or has_evidence) else 0.0

    # 5. Retry frequency
    retry_frequency = 0.0
    parse_failure_count = 0
    if trace_dir:
        pf = Path(trace_dir) / f"{doc_stem}_parse_failures.jsonl"
        entries = _parse_failure_entries(pf)
        parse_failure_count = len(entries)
        total_batches = body.get("total_batches") or 1
        retry_frequency = round(parse_failure_count / max(int(total_batches), 1), 4)

    # Optional grounding-retry stats (post-hoc targeted span extraction)
    grounding_retry_block: Dict[str, Any] = {}
    if isinstance(grounding_stats, dict):
        attempted = grounding_stats.get("retries_attempted", 0) or 0
        accepted = grounding_stats.get("retries_accepted", 0) or 0
        grounding_retry_block = {
            "grounding_retries_attempted": int(attempted),
            "grounding_retries_accepted": int(accepted),
        }

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
        **grounding_retry_block,
        **(_compute_gt_metrics(data_payload, ground_truth) if ground_truth and not failed else {}),
    }
