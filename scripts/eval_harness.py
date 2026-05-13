#!/usr/bin/env python3
"""
Multi-Document Evaluation Harness
===================================
Runs the extraction pipeline on N documents, collects per-doc scorecards,
and aggregates into a single eval_report.json + eval_summary.csv.

Failure taxonomy (documented in paper):
  F1  parse_failure       — model output was not JSON-parseable after retries
  F2  schema_mismatch     — required fields missing in extracted output
  F3  empty_field         — non-empty extraction but required field is null/empty
  F4  routing_error       — heuristic routed to wrong family (empty fields in family)
  F5  pipeline_failure    — Python exception during pipeline execution

Usage examples
--------------
  # Run on 30 Murr datasheets, domain routing, save traces
  python scripts/eval_harness.py \\
      --pdf_dir /path/to/Murr_pdfs \\
      --n_docs 30 \\
      --schema_mode domain \\
      --output_dir output/eval_murr_30 \\
      --save_debug_traces

  # Ablation condition A0: no routing, no projection
  python scripts/eval_harness.py \\
      --pdf_dir /path/to/Murr_pdfs \\
      --n_docs 30 \\
      --schema_mode auto \\
      --no_routing --no_projection \\
      --output_dir output/eval_A0

  # With ground-truth comparison (requires data/ground_truth/annotations.jsonl)
  python scripts/eval_harness.py \\
      --pdf_dir /path/to/Murr_pdfs \\
      --n_docs 30 \\
      --schema_mode domain \\
      --ground_truth data/ground_truth/annotations.jsonl \\
      --output_dir output/eval_gt_compare
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_harness")

# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

FAILURE_CLASSES: Dict[str, str] = {
    "F1": "parse_failure",       # Model output not parseable
    "F2": "schema_mismatch",     # Required fields missing (structure wrong)
    "F3": "empty_field",         # Required field extracted but null/empty
    "F4": "routing_error",       # Routing selected wrong family → empty fields
    "F5": "pipeline_failure",    # Python exception
}


def classify_failures(scorecard: Dict[str, Any]) -> List[str]:
    """
    Map a per-doc scorecard to one or more failure class codes.
    Returns [] for a fully successful extraction.
    """
    failures: List[str] = []
    status = scorecard.get("status", "unknown")

    if status == "pipeline_error":
        failures.append("F5")
        return failures

    parse_failures = scorecard.get("parse_failure_count", 0)
    if parse_failures and int(parse_failures) > 0:
        failures.append("F1")

    required_missing = scorecard.get("required_missing", [])
    if required_missing:
        # Distinguish schema mismatch (all required missing = routing issue)
        # from partial empty fields
        required_total = scorecard.get("required_total", 0)
        if required_total and len(required_missing) == required_total:
            failures.append("F4")
        elif len(required_missing) > 0:
            failures.append("F3")

    if scorecard.get("schema_validity_rate", 1.0) == 0.0 and "F3" not in failures and "F4" not in failures:
        failures.append("F2")

    return failures


# ---------------------------------------------------------------------------
# Single-doc runner: calls run_v3.py as a subprocess
# ---------------------------------------------------------------------------

def run_single_doc(
    pdf_path: str,
    output_base: str,
    schema_mode: str,
    extractor_model: str,
    extra_args: Optional[List[str]] = None,
    timeout_s: int = 600,
) -> Dict[str, Any]:
    """
    Run the full pipeline on one PDF via subprocess.

    Returns a scorecard dict with at minimum:
        doc, status, elapsed_s, failure_classes
    plus all fields from evaluate_extraction() if --evaluate succeeded.
    """
    doc_stem = Path(pdf_path).stem
    doc_output = str(Path(output_base) / doc_stem)
    Path(doc_output).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "run_v3.py"),
        pdf_path,
        "--output_dir", doc_output,
        "--extract",
        "--evaluate",
        "--schema_mode", schema_mode,
        "--extractor_model", extractor_model,
        "--save_debug_traces",
    ] + (extra_args or [])

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = round(time.time() - start, 1)
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        elapsed = timeout_s
        logger.error("  TIMEOUT: %s after %ds", doc_stem, timeout_s)
        return {
            "doc": doc_stem,
            "status": "pipeline_error",
            "error": f"timeout after {timeout_s}s",
            "elapsed_s": elapsed,
            "failure_classes": ["F5"],
        }
    except Exception as exc:
        elapsed = round(time.time() - start, 1)
        logger.error("  ERROR: %s — %s", doc_stem, exc)
        return {
            "doc": doc_stem,
            "status": "pipeline_error",
            "error": str(exc),
            "elapsed_s": elapsed,
            "failure_classes": ["F5"],
        }

    # Read evaluation artifact written by run_v3.py
    eval_path = Path(doc_output) / f"{doc_stem}_evaluation.json"
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            scorecard = json.load(f)
        scorecard["elapsed_s"] = elapsed
        scorecard["exit_code"] = exit_code
        scorecard["failure_classes"] = classify_failures(scorecard)
        return scorecard
    else:
        # Pipeline ran but evaluation artifact missing
        return {
            "doc": doc_stem,
            "status": "pipeline_error",
            "error": "evaluation artifact not found",
            "exit_code": exit_code,
            "elapsed_s": elapsed,
            "failure_classes": ["F5"],
            "stderr_tail": result.stderr[-500:] if hasattr(result, "stderr") else "",
        }


# ---------------------------------------------------------------------------
# Ground-truth comparison
# ---------------------------------------------------------------------------

def _load_ground_truth(gt_path: str) -> Dict[str, Dict[str, Any]]:
    """Load annotations.jsonl → dict keyed by doc_id."""
    gt: Dict[str, Dict[str, Any]] = {}
    if not gt_path or not Path(gt_path).exists():
        return gt
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                doc_id = record.get("doc_id") or record.get("doc")
                if doc_id:
                    gt[str(doc_id)] = record.get("fields", {})
            except Exception:
                continue
    return gt


def _field_f1_against_gt(
    extraction: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute precision, recall, and field-level F1 against ground truth.

    Matching rule: extracted value is 'correct' if the ground-truth value
    appears as a substring of the extracted value (case-insensitive), or
    vice versa.  This is intentionally lenient to handle minor formatting
    differences (e.g., "ZIM Integrated" matching "ZIM Integrated Shipping
    Services Ltd.").

    Returns a dict with: precision, recall, f1, matched_fields,
    missing_fields, extra_fields.
    """
    gt_fields = set(ground_truth.keys())
    ext_fields = set(k for k, v in extraction.items() if v not in (None, "", [], {}))

    matched: List[str] = []
    for field in gt_fields:
        gt_val = str(ground_truth.get(field, "")).lower().strip()
        # Handle FieldValue envelope
        ext_raw = extraction.get(field)
        if isinstance(ext_raw, dict):
            ext_val = str(ext_raw.get("normalized_value") or ext_raw.get("raw_value", "")).lower().strip()
        else:
            ext_val = str(ext_raw or "").lower().strip()

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


def _enrich_with_gt(
    scorecard: Dict[str, Any],
    extraction_dir: str,
    ground_truth: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Load extraction result and compute F1 against ground truth if available."""
    doc = scorecard.get("doc", "")
    gt = ground_truth.get(doc)
    if not gt:
        return scorecard

    # Load extraction result
    ext_path = Path(extraction_dir) / doc / f"{doc}_extraction_result.json"
    if not ext_path.exists():
        return scorecard

    with open(ext_path, encoding="utf-8") as f:
        extraction = json.load(f)

    if extraction.get("status") == "failed":
        scorecard["gt_field_f1"] = 0.0
        return scorecard

    f1_stats = _field_f1_against_gt(extraction, gt)
    scorecard.update(f1_stats)
    return scorecard


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_NUMERIC_METRICS = [
    "schema_validity_rate",
    "required_field_completion_rate",
    "non_empty_extraction_rate",
    "grounded_reference_rate",
    "retry_frequency",
    "elapsed_s",
    "gt_field_f1",
    "gt_field_precision",
    "gt_field_recall",
]


def aggregate_scorecards(scorecards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean/std/min/max for numeric metrics over all docs."""
    n = len(scorecards)
    success_count = sum(1 for s in scorecards if s.get("status") == "success")
    failure_counts: Dict[str, int] = {code: 0 for code in FAILURE_CLASSES}
    for s in scorecards:
        for code in s.get("failure_classes", []):
            if code in failure_counts:
                failure_counts[code] += 1

    agg: Dict[str, Any] = {
        "n_docs": n,
        "n_success": success_count,
        "n_failure": n - success_count,
        "success_rate": round(success_count / n, 4) if n > 0 else 0.0,
        "failure_breakdown": {
            code: {"label": FAILURE_CLASSES[code], "count": cnt}
            for code, cnt in failure_counts.items()
        },
    }

    for metric in _NUMERIC_METRICS:
        values = [s[metric] for s in scorecards if isinstance(s.get(metric), (int, float))]
        if values:
            agg[metric] = {
                "mean": round(statistics.mean(values), 4),
                "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "n": len(values),
            }

    return agg


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(scorecards: List[Dict[str, Any]], path: str) -> None:
    if not scorecards:
        return
    all_keys: List[str] = []
    seen: set = set()
    for s in scorecards:
        for k in s.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scorecards)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-document extraction evaluation harness")
    parser.add_argument("--pdf_dir", type=str, help="Directory of PDFs to evaluate")
    parser.add_argument("--pdf_list", type=str, help="Newline-separated file of PDF paths")
    parser.add_argument("--n_docs", type=int, default=None, help="Limit number of docs (random sample if set)")
    parser.add_argument("--schema_mode", type=str, choices=["auto", "domain", "explicit"], default="domain")
    parser.add_argument("--extractor_model", type=str, default="RMunshi/librarian-qwen-extractor")
    parser.add_argument("--output_dir", type=str, default="output/eval_run")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Path to annotations.jsonl for field-F1 evaluation")
    parser.add_argument("--timeout", type=int, default=600, help="Per-doc timeout in seconds")
    parser.add_argument("--no_routing", action="store_true", help="Ablation: bypass routing")
    parser.add_argument("--no_projection", action="store_true", help="Ablation: skip projection")
    parser.add_argument("--with_normalization", action="store_true", help="Ablation: apply FieldNormalizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect PDFs
    pdf_paths: List[str] = []
    if args.pdf_dir:
        pdf_paths = sorted(str(p) for p in Path(args.pdf_dir).glob("*.pdf"))
    elif args.pdf_list:
        with open(args.pdf_list, encoding="utf-8") as f:
            pdf_paths = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Provide --pdf_dir or --pdf_list")

    if args.n_docs and args.n_docs < len(pdf_paths):
        import random
        random.seed(args.seed)
        pdf_paths = random.sample(pdf_paths, args.n_docs)

    logger.info("Evaluating %d documents | schema_mode=%s", len(pdf_paths), args.schema_mode)
    logger.info("Output: %s", output_dir)

    # Build extra_args list
    extra_args: List[str] = []
    if args.no_routing:
        extra_args.append("--no_routing")
    if args.no_projection:
        extra_args.append("--no_projection")
    if args.with_normalization:
        extra_args.append("--with_normalization")

    ground_truth = _load_ground_truth(args.ground_truth) if args.ground_truth else {}
    if ground_truth:
        logger.info("Loaded ground truth for %d documents", len(ground_truth))

    # Run docs sequentially (GPU is shared; parallel would OOM)
    scorecards: List[Dict[str, Any]] = []
    for idx, pdf_path in enumerate(pdf_paths):
        doc_stem = Path(pdf_path).stem
        logger.info("[%d/%d] %s", idx + 1, len(pdf_paths), doc_stem)
        sc = run_single_doc(
            pdf_path=pdf_path,
            output_base=str(output_dir / "docs"),
            schema_mode=args.schema_mode,
            extractor_model=args.extractor_model,
            extra_args=extra_args,
            timeout_s=args.timeout,
        )
        if ground_truth:
            sc = _enrich_with_gt(sc, str(output_dir / "docs"), ground_truth)
        scorecards.append(sc)
        logger.info("  → status=%s  classes=%s  elapsed=%.1fs",
                    sc.get("status"), sc.get("failure_classes", []), sc.get("elapsed_s", 0))

    # Aggregate
    summary = aggregate_scorecards(scorecards)

    # Write outputs
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_doc": scorecards}, f, indent=2, ensure_ascii=False)
    logger.info("Report written → %s", report_path)

    csv_path = output_dir / "eval_summary.csv"
    _write_csv(scorecards, str(csv_path))
    logger.info("CSV written → %s", csv_path)

    # Print summary table to stdout
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY  ({args.output_dir})")
    print("=" * 70)
    print(f"  Documents   : {summary['n_docs']}  (success={summary['n_success']}, fail={summary['n_failure']})")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    for metric in _NUMERIC_METRICS:
        if metric in summary:
            m = summary[metric]
            print(f"  {metric:<38} mean={m['mean']:.4f}  std={m['std']:.4f}  [{m['min']:.4f}, {m['max']:.4f}]")
    if summary.get("failure_breakdown"):
        print("\n  Failure breakdown:")
        for code, info in summary["failure_breakdown"].items():
            if info["count"] > 0:
                print(f"    {code} {info['label']}: {info['count']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
