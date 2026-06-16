"""Calibration metrics for the grounding confidence score (task #54).

The grounding verifier emits a per-leaf ``field_confidence`` in [0, 1] that is
meant to estimate *P(the value is actually supported by the source)*. This
module answers the reliability question that estimate invites: when the verifier
says 0.8, is the value really supported ~80% of the time?

Pure stdlib, no model/IO dependency, so the math is unit-tested in isolation
against synthetic known-answer sets. The data-pairing (which leaf got which
confidence, and its hand label) lives in ``scripts/build_ece.py``.

Definitions
-----------
Given pairs ``(confidence p_i, correct y_i in {0,1})`` and M equal-width bins
over [0, 1]:

    acc(b)   = mean y_i  over leaves in bin b
    conf(b)  = mean p_i  over leaves in bin b
    ECE      = sum_b (n_b / N) * |acc(b) - conf(b)|        (lower is better)
    MCE      = max_b |acc(b) - conf(b)|                    (worst bin)
    Brier    = mean (p_i - y_i)^2                           (proper scoring rule)

ECE/MCE are binning-dependent (a known weakness); Brier is binning-free and is
reported alongside as a robustness check.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Pair = Tuple[float, bool]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def reliability_bins(
    pairs: Sequence[Pair], n_bins: int = 10
) -> List[Dict[str, float]]:
    """Equal-width reliability bins over [0, 1].

    Each bin spans ``[lo, hi)`` except the last, which is closed ``[lo, 1.0]`` so
    a perfect confidence of 1.0 lands in the top bin rather than overflowing.
    Empty bins are returned with ``count == 0`` (they contribute 0 to the ECE
    weight) so the diagram shows the gaps honestly.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = [i / n_bins for i in range(n_bins + 1)]
    bins: List[Dict[str, float]] = [
        {
            "lo": edges[i],
            "hi": edges[i + 1],
            "count": 0,
            "sum_conf": 0.0,
            "sum_correct": 0.0,
        }
        for i in range(n_bins)
    ]

    for conf, correct in pairs:
        p = _clamp01(conf)
        # idx = floor(p * n_bins), with p == 1.0 folded into the last bin.
        idx = int(p * n_bins)
        if idx >= n_bins:
            idx = n_bins - 1
        b = bins[idx]
        b["count"] += 1
        b["sum_conf"] += p
        b["sum_correct"] += 1.0 if correct else 0.0

    for b in bins:
        n = b["count"]
        b["avg_conf"] = (b["sum_conf"] / n) if n else None
        b["accuracy"] = (b["sum_correct"] / n) if n else None
        b["gap"] = (
            abs(b["accuracy"] - b["avg_conf"]) if n else None
        )
    return bins


def expected_calibration_error(pairs: Sequence[Pair], n_bins: int = 10) -> float:
    """Weighted mean bin gap. 0.0 for an empty set (nothing to miscalibrate)."""
    n_total = len(pairs)
    if n_total == 0:
        return 0.0
    ece = 0.0
    for b in reliability_bins(pairs, n_bins):
        if b["count"]:
            ece += (b["count"] / n_total) * b["gap"]
    return ece


def max_calibration_error(pairs: Sequence[Pair], n_bins: int = 10) -> float:
    """Largest gap over non-empty bins."""
    gaps = [b["gap"] for b in reliability_bins(pairs, n_bins) if b["count"]]
    return max(gaps) if gaps else 0.0


def brier_score(pairs: Sequence[Pair]) -> float:
    """Mean squared error of confidence vs. outcome. Binning-free."""
    if not pairs:
        return 0.0
    return sum((_clamp01(p) - (1.0 if y else 0.0)) ** 2 for p, y in pairs) / len(pairs)


def calibration_report(pairs: Sequence[Pair], n_bins: int = 10) -> Dict[str, object]:
    """Full report: scalar metrics + per-bin table + a base-rate sanity line."""
    n = len(pairs)
    base_rate = (sum(1 for _, y in pairs if y) / n) if n else None
    mean_conf = (sum(_clamp01(p) for p, _ in pairs) / n) if n else None
    raw_bins = reliability_bins(pairs, n_bins)
    bins_out = []
    for i, b in enumerate(raw_bins):
        closer = "]" if i == n_bins - 1 else ")"
        bins_out.append(
            {
                "range": f"[{b['lo']:.2f},{b['hi']:.2f}{closer}",
                "count": b["count"],
                "avg_conf": b["avg_conf"],
                "accuracy": b["accuracy"],
                "gap": b["gap"],
            }
        )
    return {
        "n": n,
        "n_bins": n_bins,
        "base_rate": base_rate,            # actual fraction supported
        "mean_confidence": mean_conf,      # mean predicted — vs base_rate = over/under-confidence
        "ece": expected_calibration_error(pairs, n_bins),
        "mce": max_calibration_error(pairs, n_bins),
        "brier": brier_score(pairs),
        "bins": bins_out,
    }


def render_reliability_table(report: Dict[str, object]) -> str:
    """Markdown table of the per-bin reliability diagram."""
    lines = [
        "| confidence bin | n | avg conf | accuracy | gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for b in report["bins"]:  # type: ignore[index]
        if not b["count"]:
            lines.append(f"| {b['range']} | 0 | — | — | — |")
            continue
        lines.append(
            f"| {b['range']} | {b['count']} | {b['avg_conf']:.3f} "
            f"| {b['accuracy']:.3f} | {b['gap']:.3f} |"
        )
    return "\n".join(lines)
