"""Known-answer tests for the calibration math (task #54).

These pin ECE/MCE/Brier on synthetic sets whose answers are computable by hand,
so the metric is trustworthy before it is ever pointed at real grounding data.
"""
from extractor.calibration import (
    reliability_bins,
    expected_calibration_error,
    max_calibration_error,
    brier_score,
    calibration_report,
    render_reliability_table,
)


def _approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_perfectly_calibrated_is_zero_ece():
    # Accuracy equals the confidence in every populated bin, so every gap is 0:
    # conf 0.0 -> 0% correct, conf 1.0 -> 100% correct, conf 0.5 -> half correct.
    pairs = (
        [(0.0, False)] * 10
        + [(1.0, True)] * 10
        + [(0.5, True)] * 5 + [(0.5, False)] * 5
    )
    ece = expected_calibration_error(pairs, n_bins=10)
    assert _approx(ece, 0.0), ece


def test_overconfident_set_has_large_ece_and_mce():
    # Always says 1.0 but is only right half the time -> gap of ~0.5 in the top bin.
    pairs = [(1.0, True)] * 50 + [(1.0, False)] * 50
    ece = expected_calibration_error(pairs, n_bins=10)
    mce = max_calibration_error(pairs, n_bins=10)
    assert _approx(ece, 0.5, tol=1e-9), ece   # |0.5 acc - 1.0 conf| * (100/100)
    assert _approx(mce, 0.5, tol=1e-9), mce


def test_underconfident_set_also_penalised():
    # Says 0.0 but is always right -> full gap, ECE = 1.0.
    pairs = [(0.0, True)] * 20
    assert _approx(expected_calibration_error(pairs, n_bins=10), 1.0)


def test_brier_known_values():
    # All confident-and-correct -> Brier 0.
    assert _approx(brier_score([(1.0, True)] * 5), 0.0)
    # Confidence 0.5 everywhere -> (0.5)^2 = 0.25 regardless of outcome.
    assert _approx(brier_score([(0.5, True), (0.5, False)]), 0.25)
    # Confident-and-wrong -> (1-0)^2 = 1.0.
    assert _approx(brier_score([(1.0, False)] * 3), 1.0)


def test_confidence_of_one_lands_in_last_bin_not_overflow():
    bins = reliability_bins([(1.0, True)], n_bins=10)
    assert bins[-1]["count"] == 1, "p==1.0 must fold into the closed top bin"
    assert sum(b["count"] for b in bins) == 1


def test_empty_bins_reported_with_zero_count():
    # Only the top bin is populated; the other nine must still be present, empty.
    bins = reliability_bins([(0.95, True)] * 4, n_bins=10)
    assert len(bins) == 10
    empties = [b for b in bins if b["count"] == 0]
    assert len(empties) == 9
    assert all(b["avg_conf"] is None and b["accuracy"] is None for b in empties)


def test_empty_input_is_zero_not_crash():
    assert expected_calibration_error([], 10) == 0.0
    assert max_calibration_error([], 10) == 0.0
    assert brier_score([]) == 0.0
    rep = calibration_report([], 10)
    assert rep["n"] == 0 and rep["base_rate"] is None


def test_out_of_range_confidence_is_clamped():
    # A stray >1 or <0 must not escape its bin or distort Brier.
    bins = reliability_bins([(1.4, True), (-0.3, False)], n_bins=10)
    assert bins[-1]["count"] == 1 and bins[0]["count"] == 1
    assert _approx(brier_score([(1.4, True)]), 0.0)  # clamps to 1.0 -> correct


def test_report_and_table_render():
    pairs = [(0.9, True)] * 8 + [(0.9, False)] * 2 + [(0.2, False)] * 10
    rep = calibration_report(pairs, n_bins=10)
    assert rep["n"] == 20
    assert _approx(rep["base_rate"], 8 / 20)
    # bin .9-1.0: conf .9, acc .8 -> gap .1, weight 10/20 -> .05
    # bin .2-.3: conf .2, acc 0  -> gap .2, weight 10/20 -> .10  => ECE .15
    assert _approx(rep["ece"], 0.15, tol=1e-9), rep["ece"]
    table = render_reliability_table(rep)
    assert "confidence bin" in table and "0.90,1.00]" in table
