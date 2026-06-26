"""Task #55 — broadened grounding normalization recovers notation-only drift.

Calibration (#54) showed the verifier scored values 0.0 despite being present,
because the source rendered the same datum with different notation: decimal
comma vs dot, '±' vs '+/-', '5 %' vs '5%', and scale words ('2 Mio.' for
2000000). These pin that those forms now ground, while genuinely-absent or
fabricated values still flag (the fold must be value-preserving).
"""
from extractor.agent import ExtractorAgent as A


def _verify(record, source):
    return A._verify_string_spans(record, source)


def test_decimal_comma_value_grounds_against_dot_source():
    # Emitted '4.3', source writes '4,3 mm'. Same datum, different separator.
    st = _verify({"v": "4.3 mm"}, "Outer diameter (jacket) 4,3 mm Tolerance")
    assert st["field_confidence"]["v"] == 1.0
    assert st["verified"] == 1 and not st["flagged"]


def test_plus_minus_symbol_grounds_against_ascii():
    st = _verify({"tol": "+/- 5%"}, "core insulation tolerance ± 5 % nominal")
    assert st["field_confidence"]["tol"] == 1.0
    assert not st["flagged"]


def test_percent_spacing_is_folded():
    st = _verify({"v": "value 5%"}, "the value 5 % of nominal")
    assert st["field_confidence"]["v"] == 1.0


def test_scale_word_million_grounds_numeric_leaf():
    # Numeric leaf 2000000 must match source '2 Mio.' (and '5 million').
    st = _verify(
        {"row": "Torsion cycles", "min_value": 2000000},
        "No. of torsion cycles 2 Mio. Torsion stress",
    )
    assert st["field_confidence"]["min_value"] == 1.0
    assert st["verified"] >= 1


def test_scale_word_does_not_trip_on_millimeter():
    # '99 millimeter' must NOT be read as 99,000,000 — the word-boundary guard.
    # If the guard failed, 99000000 would ground at 1.0; it must not.
    st = _verify(
        {"row": "Cable length", "len": 99000000},
        "Cable length 99 millimeter jacket PUR",
    )
    assert st["field_confidence"]["len"] < 1.0, "millimeter wrongly scaled to millions"


def test_fold_is_value_preserving_no_false_verbatim_match():
    # The comma->dot fold must equate NOTATIONS, never different VALUES: emitted
    # '7,7 mm' must not VERBATIM-ground (1.0) against source '3,2 mm'. (A lower
    # fuzzy-snap score is the pre-existing #51 repair, separate from this fold.)
    st = _verify({"v": "thickness 7,7 mm"}, "sheath thickness 3,2 mm nominal")
    assert st["field_confidence"]["v"] < 1.0, "fold falsely equated 7,7 and 3,2"


def test_dissimilar_value_still_flags():
    # A value absent in any notation, with low overlap, must remain flagged —
    # the broadened normalize must not invent grounding.
    st = _verify({"v": "rated current 7,7 amps maximum"}, "operating voltage thirty volts dc only")
    assert st["field_confidence"]["v"] < 0.85
    assert any(f["path"] == "v" for f in st["flagged"])


def test_normal_verbatim_still_grounds():
    st = _verify({"v": "Stripping length (jacket)"}, "Side 2 Stripping length (jacket) 20 mm")
    assert st["field_confidence"]["v"] == 1.0
