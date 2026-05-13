#!/usr/bin/env python3
"""
Ablation Study Runner
======================
Runs 4 controlled conditions on the same document set.
Each condition controls which pipeline stages are active:

  A0  baseline          — no routing, no projection, no normalization
  A1  routing           — heuristic routing on; no projection; no normalization
  A2  routing+proj      — routing + project_to_schema(); no normalization
  A3  full              — routing + projection + FieldNormalizer (complete system)

For each condition the script calls eval_harness.py and writes its output
to ablation/<condition_id>/.

After all conditions complete, a comparison table is written to
ablation/ablation_comparison.json and printed to stdout.

Usage
-----
  python scripts/run_ablation.py \\
      --pdf_dir /path/to/Murr_pdfs \\
      --n_docs 30 \\
      --extractor_model RMunshi/librarian-qwen-extractor \\
      --output_dir output/ablation_run

  # To re-run only specific conditions:
  python scripts/run_ablation.py \\
      --pdf_dir /path/to/Murr_pdfs \\
      --n_docs 30 \\
      --conditions A0 A3 \\
      --output_dir output/ablation_run
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ablation")

# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

CONDITIONS: Dict[str, Dict[str, Any]] = {
    "A0_baseline": {
        "label": "Baseline (no routing, no projection, no normalization)",
        "schema_mode": "auto",
        "extra_flags": ["--no_routing", "--no_projection"],
    },
    "A1_routing": {
        "label": "Routing only",
        "schema_mode": "domain",
        "extra_flags": [],
        # projection is skipped via no_projection flag because we want
        # to isolate the routing contribution before adding projection
        "extra_flags": ["--no_projection"],
    },
    "A2_routing_projection": {
        "label": "Routing + projection",
        "schema_mode": "domain",
        "extra_flags": [],
    },
    "A3_full": {
        "label": "Full system (routing + projection + normalization)",
        "schema_mode": "domain",
        "extra_flags": ["--with_normalization"],
    },
}

# Metrics compared across conditions (must match keys in eval_report.json summary)
_COMPARE_METRICS = [
    "success_rate",
    "schema_validity_rate",
    "required_field_completion_rate",
    "non_empty_extraction_rate",
    "grounded_reference_rate",
    "retry_frequency",
    "gt_field_f1",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_condition(
    condition_id: str,
    condition: Dict[str, Any],
    pdf_dir: Optional[str],
    pdf_list: Optional[str],
    n_docs: int,
    extractor_model: str,
    output_base: str,
    ground_truth: Optional[str],
    timeout: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Run eval_harness.py for one condition and return the summary dict."""
    cond_output = str(Path(output_base) / condition_id)
    harness = str(Path(__file__).parent / "eval_harness.py")

    cmd = [
        sys.executable,
        harness,
        "--n_docs", str(n_docs),
        "--schema_mode", condition["schema_mode"],
        "--extractor_model", extractor_model,
        "--output_dir", cond_output,
        "--timeout", str(timeout),
        "--seed", str(seed),
    ] + condition.get("extra_flags", [])

    if pdf_dir:
        cmd += ["--pdf_dir", pdf_dir]
    elif pdf_list:
        cmd += ["--pdf_list", pdf_list]

    if ground_truth:
        cmd += ["--ground_truth", ground_truth]

    logger.info("=" * 60)
    logger.info("Running condition: %s", condition_id)
    logger.info("  %s", condition["label"])
    logger.info("  cmd: %s", " ".join(cmd))

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.error("Condition %s failed with exit code %d", condition_id, result.returncode)
        return None

    report_path = Path(cond_output) / "eval_report.json"
    if not report_path.exists():
        logger.error("Report not found for condition %s: %s", condition_id, report_path)
        return None

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    return report.get("summary", {})


def _build_comparison_table(
    condition_summaries: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a structured comparison dict keyed by metric then condition."""
    table: Dict[str, Any] = {}

    for metric in _COMPARE_METRICS:
        row: Dict[str, Any] = {}
        for cid, summary in condition_summaries.items():
            if summary is None:
                row[cid] = None
                continue
            # Scalar (e.g. success_rate) or nested {mean, std, ...}
            val = summary.get(metric)
            if isinstance(val, dict):
                row[cid] = val.get("mean")
            else:
                row[cid] = val
        table[metric] = row

    return table


def _print_comparison_table(
    table: Dict[str, Any],
    condition_summaries: Dict[str, Optional[Dict[str, Any]]],
) -> None:
    condition_ids = list(condition_summaries.keys())
    col_w = 22
    header = f"{'Metric':<38}" + "".join(f"{cid:<{col_w}}" for cid in condition_ids)
    print("\n" + "=" * (38 + col_w * len(condition_ids)))
    print("ABLATION COMPARISON TABLE")
    print("=" * (38 + col_w * len(condition_ids)))
    print(header)
    print("-" * (38 + col_w * len(condition_ids)))

    for metric, row in table.items():
        line = f"{metric:<38}"
        for cid in condition_ids:
            val = row.get(cid)
            cell = f"{val:.4f}" if isinstance(val, float) else str(val)
            line += f"{cell:<{col_w}}"
        print(line)

    print("=" * (38 + col_w * len(condition_ids)) + "\n")

    # Print condition labels legend
    for cid, summary in condition_summaries.items():
        label = CONDITIONS.get(cid, {}).get("label", cid)
        n = summary.get("n_docs", "?") if summary else "?"
        print(f"  {cid}: {label}  (n={n})")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study: 4-condition evaluation")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--pdf_dir", type=str, help="Directory of PDFs")
    data_group.add_argument("--pdf_list", type=str, help="File with PDF paths (one per line)")
    parser.add_argument("--n_docs", type=int, default=30, help="Number of docs per condition")
    parser.add_argument("--extractor_model", type=str, default="RMunshi/librarian-qwen-extractor")
    parser.add_argument("--output_dir", type=str, default="output/ablation")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Path to annotations.jsonl for field-F1 evaluation")
    parser.add_argument("--timeout", type=int, default=600, help="Per-doc timeout in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Run only these condition IDs (e.g. A0_baseline A3_full)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = args.conditions or list(CONDITIONS.keys())
    invalid = [c for c in selected if c not in CONDITIONS]
    if invalid:
        parser.error(f"Unknown condition(s): {invalid}. Valid: {list(CONDITIONS.keys())}")

    condition_summaries: Dict[str, Optional[Dict[str, Any]]] = {}

    for condition_id in selected:
        summary = _run_condition(
            condition_id=condition_id,
            condition=CONDITIONS[condition_id],
            pdf_dir=args.pdf_dir,
            pdf_list=args.pdf_list,
            n_docs=args.n_docs,
            extractor_model=args.extractor_model,
            output_base=str(output_dir),
            ground_truth=args.ground_truth,
            timeout=args.timeout,
            seed=args.seed,
        )
        condition_summaries[condition_id] = summary

    # Build and write comparison table
    comparison = _build_comparison_table(condition_summaries)

    comparison_report = {
        "conditions": {
            cid: {
                "label": CONDITIONS[cid]["label"],
                "summary": condition_summaries[cid],
            }
            for cid in selected
        },
        "comparison_table": comparison,
    }

    report_path = output_dir / "ablation_comparison.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2, ensure_ascii=False)
    logger.info("Ablation comparison written → %s", report_path)

    _print_comparison_table(comparison, condition_summaries)


if __name__ == "__main__":
    main()
