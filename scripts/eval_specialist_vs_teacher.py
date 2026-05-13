#!/usr/bin/env python3
"""
Specialist vs Teacher Evaluation
==================================
Compares the distilled specialist model (Qwen local) against the teacher
model (GPT-4o or equivalent) on the same document set.

This produces the *distillation effectiveness* section of the paper:
"our specialist achieves X% of teacher performance at 1/20th the cost."

What is measured
----------------
For each document, both models are run via run_v3.py with identical
schema_mode, projection, and normalization settings.  The only variable
is --extractor_model.

Metrics computed (from eval_harness summary per model):
  - non_empty_extraction_rate
  - required_field_completion_rate
  - schema_validity_rate
  - gt_field_f1  (if --ground_truth provided)

The specialist's score as a percentage of the teacher's score is the
*retention rate* — the fraction of teacher quality preserved after
distillation.

Usage
-----
  python scripts/eval_specialist_vs_teacher.py \\
      --pdf_dir /path/to/pdfs \\
      --n_docs 30 \\
      --specialist_model RMunshi/librarian-qwen-extractor \\
      --teacher_model gpt-4o \\
      --output_dir output/distillation_eval \\
      --ground_truth data/ground_truth/annotations.jsonl

  # Use a local specialist path
  python scripts/eval_specialist_vs_teacher.py \\
      --pdf_dir /path/to/pdfs \\
      --n_docs 30 \\
      --specialist_model research/librarian_qwen_specialist \\
      --teacher_model gpt-4o \\
      --output_dir output/distillation_eval
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
logger = logging.getLogger("eval_specialist_vs_teacher")

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

def _run_eval(
    model_id: str,
    label: str,
    pdf_dir: Optional[str],
    pdf_list: Optional[str],
    n_docs: int,
    output_dir: str,
    schema_mode: str,
    ground_truth: Optional[str],
    timeout: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Run eval_harness.py for one model and return the summary."""
    harness = str(Path(__file__).parent / "eval_harness.py")
    cmd = [
        sys.executable,
        harness,
        "--n_docs", str(n_docs),
        "--schema_mode", schema_mode,
        "--extractor_model", model_id,
        "--output_dir", output_dir,
        "--timeout", str(timeout),
        "--seed", str(seed),
        "--with_normalization",
    ]
    if pdf_dir:
        cmd += ["--pdf_dir", pdf_dir]
    elif pdf_list:
        cmd += ["--pdf_list", pdf_list]
    if ground_truth:
        cmd += ["--ground_truth", ground_truth]

    logger.info("Running %s (%s)...", label, model_id)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.error("%s evaluation failed (exit %d)", label, result.returncode)
        return None

    report_path = Path(output_dir) / "eval_report.json"
    if not report_path.exists():
        logger.error("Report not found: %s", report_path)
        return None

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    return report.get("summary", {})


def _scalar(summary: Dict[str, Any], metric: str) -> Optional[float]:
    """Extract scalar value from summary dict (handles nested mean/std dicts)."""
    if summary is None:
        return None
    val = summary.get(metric)
    if isinstance(val, dict):
        return val.get("mean")
    return val


def _retention_rate(teacher_val: Optional[float], specialist_val: Optional[float]) -> Optional[float]:
    """Specialist score / teacher score, clamped to [0, 1]."""
    if teacher_val is None or specialist_val is None:
        return None
    if teacher_val == 0.0:
        return None
    return round(min(specialist_val / teacher_val, 1.0), 4)


def _print_comparison(
    teacher_summary: Dict[str, Any],
    specialist_summary: Dict[str, Any],
    teacher_label: str,
    specialist_label: str,
) -> None:
    col_w = 18
    header = f"{'Metric':<38}{teacher_label:<{col_w}}{specialist_label:<{col_w}}{'Retention %':<{col_w}}"
    print("\n" + "=" * (38 + col_w * 3))
    print("SPECIALIST vs TEACHER COMPARISON")
    print("=" * (38 + col_w * 3))
    print(header)
    print("-" * (38 + col_w * 3))

    for metric in _COMPARE_METRICS:
        t_val = _scalar(teacher_summary, metric)
        s_val = _scalar(specialist_summary, metric)
        ret = _retention_rate(t_val, s_val)

        t_str = f"{t_val:.4f}" if t_val is not None else "n/a"
        s_str = f"{s_val:.4f}" if s_val is not None else "n/a"
        r_str = f"{ret * 100:.1f}%" if ret is not None else "n/a"
        print(f"{metric:<38}{t_str:<{col_w}}{s_str:<{col_w}}{r_str:<{col_w}}")

    print("=" * (38 + col_w * 3) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Specialist vs teacher distillation evaluation")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--pdf_dir", type=str)
    data_group.add_argument("--pdf_list", type=str)
    parser.add_argument("--n_docs", type=int, default=30)
    parser.add_argument("--specialist_model", type=str, default="RMunshi/librarian-qwen-extractor")
    parser.add_argument("--teacher_model", type=str, default="gpt-4o")
    parser.add_argument("--schema_mode", type=str, default="domain")
    parser.add_argument("--output_dir", type=str, default="output/distillation_eval")
    parser.add_argument("--ground_truth", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "teacher").mkdir(parents=True, exist_ok=True)
    (output_dir / "specialist").mkdir(parents=True, exist_ok=True)

    teacher_summary = _run_eval(
        model_id=args.teacher_model,
        label="teacher",
        pdf_dir=args.pdf_dir,
        pdf_list=args.pdf_list,
        n_docs=args.n_docs,
        output_dir=str(output_dir / "teacher"),
        schema_mode=args.schema_mode,
        ground_truth=args.ground_truth,
        timeout=args.timeout,
        seed=args.seed,
    )

    specialist_summary = _run_eval(
        model_id=args.specialist_model,
        label="specialist",
        pdf_dir=args.pdf_dir,
        pdf_list=args.pdf_list,
        n_docs=args.n_docs,
        output_dir=str(output_dir / "specialist"),
        schema_mode=args.schema_mode,
        ground_truth=args.ground_truth,
        timeout=args.timeout,
        seed=args.seed,
    )

    # Build comparison
    comparison: Dict[str, Any] = {}
    for metric in _COMPARE_METRICS:
        t_val = _scalar(teacher_summary, metric)
        s_val = _scalar(specialist_summary, metric)
        comparison[metric] = {
            "teacher": t_val,
            "specialist": s_val,
            "retention_rate": _retention_rate(t_val, s_val),
        }

    report = {
        "teacher_model": args.teacher_model,
        "specialist_model": args.specialist_model,
        "n_docs": args.n_docs,
        "schema_mode": args.schema_mode,
        "teacher_summary": teacher_summary,
        "specialist_summary": specialist_summary,
        "comparison": comparison,
        # Overall retention: mean of non-null retention rates
        "mean_retention_rate": round(
            sum(v["retention_rate"] for v in comparison.values() if v["retention_rate"] is not None)
            / max(sum(1 for v in comparison.values() if v["retention_rate"] is not None), 1),
            4,
        ),
    }

    report_path = output_dir / "distillation_comparison.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report written → %s", report_path)

    if teacher_summary and specialist_summary:
        _print_comparison(
            teacher_summary, specialist_summary,
            args.teacher_model, args.specialist_model,
        )
        print(f"  Mean retention rate: {report['mean_retention_rate'] * 100:.1f}%\n")

    logger.info("Done.")


if __name__ == "__main__":
    main()
