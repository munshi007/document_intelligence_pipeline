"""Behavioural tests for TSR grid normalization + unassigned-word rescue
(task #49).

Live failure: TATR's last column band on sample-invoice's header table ended
~1pt left of the word "2024", so center-containment fill dropped the year —
"1. März 2024" became "1. März" in the markdown, before the extractor ever
saw the text. The same mechanism loses any word whose center lands in a gap
between predicted bands. Two layers fix the class:

  1. `_normalize_bands` (tatr.py): dedupe duplicate band detections, snap
     adjacent boundaries to the midpoint, stretch outer bands to the crop
     bounds — the cell grid tiles the table region.
  2. `_rescue_unassigned_words` (extract_tsr.py, engine-agnostic): any word
     still unassigned after fill goes to its nearest cell instead of being
     silently dropped.
"""
from processors.tables_v2.extract_tsr import TableExtractorTSR
from processors.tables_v2.tsr.tatr import _normalize_bands
from processors.tables_v2.types import TableCell, WordSpan


def _band(b0, b1, score=0.9, axis="x"):
    bbox = [b0, 0.0, b1, 50.0] if axis == "x" else [0.0, b0, 100.0, b1]
    return {"bbox": bbox, "score": score}


# ── _normalize_bands ─────────────────────────────────────────────────


def test_outer_bands_stretch_to_crop_bounds():
    # The live bug: last column stopped at 524 while the table runs to 564.
    cols = [_band(10, 200), _band(210, 524)]
    out = _normalize_bands(cols, axis="x", lo=0.0, hi=564.0)
    assert out[0]["bbox"][0] == 0.0
    assert out[-1]["bbox"][2] == 564.0


def test_interior_gaps_close_at_midpoint():
    cols = [_band(0, 100), _band(110, 200)]
    out = _normalize_bands(cols, axis="x", lo=0.0, hi=200.0)
    assert out[0]["bbox"][2] == 105.0
    assert out[1]["bbox"][0] == 105.0


def test_overlapping_neighbours_split_at_midpoint():
    cols = [_band(0, 120), _band(100, 200)]
    out = _normalize_bands(cols, axis="x", lo=0.0, hi=200.0)
    assert out[0]["bbox"][2] == 110.0
    assert out[1]["bbox"][0] == 110.0


def test_duplicate_band_detections_are_deduped():
    # Live repro: TATR emitted cols (303..464) and (320..463) — one column
    # detected twice. The lower-scoring duplicate must be dropped, not split.
    cols = [
        _band(303, 464, score=0.95),
        _band(320, 463, score=0.60),
        _band(470, 520, score=0.90),
    ]
    out = _normalize_bands(cols, axis="x", lo=0.0, hi=564.0)
    assert len(out) == 2
    assert out[0]["bbox"][0] == 0.0          # stretched left
    assert out[0]["bbox"][2] == out[1]["bbox"][0]  # midpoint-snapped
    assert out[1]["bbox"][2] == 564.0        # stretched right


def test_rows_axis_normalizes_y():
    rows = [_band(332, 341, axis="y"), _band(343, 352, axis="y")]
    out = _normalize_bands(rows, axis="y", lo=0.0, hi=359.0)
    assert out[0]["bbox"][1] == 0.0
    assert out[0]["bbox"][3] == out[1]["bbox"][1] == 342.0
    assert out[1]["bbox"][3] == 359.0


def test_single_band_covers_full_extent():
    out = _normalize_bands([_band(40, 60)], axis="x", lo=0.0, hi=100.0)
    assert out[0]["bbox"][0] == 0.0 and out[0]["bbox"][2] == 100.0


# ── _rescue_unassigned_words ─────────────────────────────────────────


def _word(wid, text, x0, y0, x1, y1):
    return WordSpan(id=wid, text=text, bbox=(x0, y0, x1, y1))


def _cell(row, col, bbox):
    return TableCell(row=row, col=col, bbox_pdf=bbox, text="", word_ids=[])


def test_word_outside_every_cell_lands_in_nearest_cell():
    # Mirror of the live failure geometry: the date cell ends at x=520,
    # "2024" is centered at x=523 — 3pt outside.
    extractor = TableExtractorTSR(engine=None)
    cells = [
        _cell(1, 0, (100.0, 343.0, 300.0, 353.0)),
        _cell(1, 1, (460.0, 343.0, 520.0, 353.0)),
    ]
    words = [
        _word(1, "1.", 480.6, 345.1, 488.1, 355.1),
        _word(2, "März", 490.6, 345.1, 510.5, 355.1),
        _word(3, "2024", 512.9, 345.1, 533.1, 355.1),
    ]
    used = set()
    for c in cells:
        extractor._fill_cell_with_words(c, words, used)
    assert cells[1].text == "1. März", "precondition: containment drops the year"

    extractor._rescue_unassigned_words(cells, words, used)
    assert cells[1].text == "1. März 2024"
    assert cells[1].word_ids == [1, 2, 3]
    assert used == {1, 2, 3}


def test_rescue_keeps_reading_order():
    extractor = TableExtractorTSR(engine=None)
    # Rescued word belongs BEFORE an already-assigned word in reading order.
    cells = [_cell(0, 0, (50.0, 10.0, 200.0, 20.0))]
    words = [
        _word(1, "world", 100.0, 11.0, 140.0, 19.0),
        _word(2, "hello", 30.0, 11.0, 48.0, 19.0),  # left of the cell band
    ]
    used = set()
    extractor._fill_cell_with_words(cells[0], words, used)
    extractor._rescue_unassigned_words(cells, words, used)
    assert cells[0].text == "hello world"


def test_rescue_noop_when_all_words_assigned():
    extractor = TableExtractorTSR(engine=None)
    cells = [_cell(0, 0, (0.0, 0.0, 100.0, 20.0))]
    words = [_word(1, "a", 10.0, 5.0, 20.0, 15.0)]
    used = set()
    extractor._fill_cell_with_words(cells[0], words, used)
    before = cells[0].text
    extractor._rescue_unassigned_words(cells, words, used)
    assert cells[0].text == before == "a"


def test_rescue_handles_no_cells():
    extractor = TableExtractorTSR(engine=None)
    words = [_word(1, "orphan", 0.0, 0.0, 10.0, 10.0)]
    extractor._rescue_unassigned_words([], words, set())  # must not raise
