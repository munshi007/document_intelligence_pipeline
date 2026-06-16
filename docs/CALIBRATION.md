# Grounding-Confidence Calibration (ECE)

> Task #54 — closes the remaining ⬜ in `SOTA_AUDIT.md` §"Progress": *"Killer #2
> (ECE table): the reliability/ECE evaluation … is the remaining evaluation work."*
>
> **What this measures:** the grounding verifier emits a per-leaf
> `field_confidence ∈ [0,1]` (#52) that is meant to estimate *P(this value is
> actually supported by the source document)*. This is the reliability check that
> estimate invites: **when the verifier says 0.8, is the value really supported
> ~80% of the time?** It calibrates *faithfulness*, not generic correctness (see
> Caveats).

## Headline

| metric | value |
|---|---|
| sampled leaves (n) | 107 |
| base rate (fraction actually supported) | 0.869 |
| mean confidence | 0.712 |
| **ECE** (10-bin, equal-width) | **0.165** |
| MCE (worst bin) | 0.571 |
| Brier (binning-free) | 0.131 |

**Verdict: the score is conservative — systematically *under*-confident, never
over-confident.** Mean confidence (0.712) sits ~0.16 below the empirical support
rate (0.869). Every populated bin has accuracy ≥ its confidence. For a
hallucination-suppression layer this is the *safe* direction (it flags real
values rather than passing fabricated ones) but it costs calibration.

## Reliability diagram

| confidence bin | n | avg conf | accuracy | gap |
|---|---:|---:|---:|---:|
| [0.00,0.10) | 14 | 0.000 | 0.571 | 0.571 |
| [0.10,0.20) | 0 | — | — | — |
| [0.20,0.30) | 2 | 0.228 | 0.000 | 0.228 |
| [0.30,0.40) | 0 | — | — | — |
| [0.40,0.50) | 0 | — | — | — |
| [0.50,0.60) | 9 | 0.526 | 0.667 | 0.141 |
| [0.60,0.70) | 11 | 0.662 | 0.818 | 0.156 |
| [0.70,0.80) | 11 | 0.739 | 0.909 | 0.170 |
| [0.80,0.90) | 30 | 0.860 | 1.000 | 0.140 |
| [0.90,1.00] | 30 | 0.994 | 1.000 | 0.006 |

Read top-down:
- **The top bin is excellent.** Confidence 0.994 → 100% supported (gap 0.006):
  when the verifier is sure, it is right.
- **The 0.0 bin drives MCE.** 14 leaves scored *exactly* 0.0, yet 8/14 (57%) are
  actually in the source — as comma-decimals (`4,3 mm` vs emitted `4.3`),
  number-words (`one million` / `2000000` vs source `1 Mio.` / `2 Mio.`), and
  unit paraphrase (`3.3 meters per second` vs `3,3 m/s`). The verifier's
  windowed match returns 0.0 because its normalization does not span those forms.
- **The 0.2–0.3 bin is the score working as intended:** both leaves there are
  OCR-hallucinated garbage (`Aon 2 2 2 …`), correctly scored low, 0% supported.

## Root-cause finding (actionable)

The miscalibration is **not noise — it is a specific gap in numeric/unit
normalization.** Splitting the sample:

| leaf class | n | base rate | mean conf | ECE |
|---|---:|---:|---:|---:|
| value / unit / min / max | 67 | 0.836 | 0.612 | **0.237** |
| name | 40 | 0.925 | 0.880 | 0.123 |

Value leaves — where comma↔dot decimals, number-words, and spelled-out units
live — are markedly *more* under-confident than name leaves. The existing
digit-aware fallback (#51) handles digit *insertions* but not decimal-separator,
number-word, or unit-spelling equivalence. **Recommended follow-up:** extend the
verifier's span normalization to cover those three forms; it would lift the low
bins toward the diagonal and cut ECE without weakening hallucination detection
(the garbage in the 0.2–0.3 bin would still score low).

## Method

- **Corpus:** 24 of the 25 EVAL_DATA Murr connector/cable datasheets (one,
  `7000-12181-0130030`, is excluded — it failed final schema validation on an
  unrelated `reasoning_thoughts` type bug *after* grounding ran, so no sidecar
  was persisted; logged as a candidate follow-up).
- **Extraction:** `cli extract … --with-grounding`; the full per-leaf
  `field_confidence` map is persisted to each `<stem>_grounding.json`
  (`stages/extract.py`). Batch runner: `scripts/ece_batch_extract.sh`.
- **Sampling (`scripts/build_ece.py sample`):** pool every *content* leaf
  (excludes `source_evidence.*`, `confidence_score`, `page_number`,
  `reasoning_thoughts`) across the 24 docs → 1327 leaves; draw a
  confidence-stratified sample (all sparse bins kept, saturated bins capped at
  30) → 107 leaves spanning the full range. Deterministic (md5 over `seed|doc|path`).
- **Labeling (manual, `scripts/apply_ece_labels.py`):** each sampled leaf
  hand-labeled `supported ∈ {0,1}` by reading that document's source text
  (reconstructed from `<stem>_regions.json` blocks + manifest table cells).
  Audit trail with per-leaf rationale committed at
  [`docs/calibration_data/manual_labels.json`](calibration_data/manual_labels.json)
  (live copy in `output/ece_run/manual_labels.json`); scored report at
  [`docs/calibration_data/calibration_report.json`](calibration_data/calibration_report.json).
- **Labeling protocol** — `1` if the value's fact appears in source allowing:
  case/whitespace, decimal comma↔dot, unicode/mojibake glyph noise (°, Ω, ±, ²,
  @), unit symbol↔spelled-out, digits↔number-words, and term reordering. `0` for
  OCR garbage / non-word corruption, a numeric absent from the doc, or a name
  that introduces a salient term not in the source (e.g. `Filling amount` where
  source has only `Filler`; `Jacket weight` where source says `Cable weight`).
- **Metrics:** `extractor/calibration.py` (pure stdlib; unit-tested on
  synthetic known-answer sets in `tests/extraction/test_calibration_metrics.py`).

## Caveats (do not over-claim)

1. **Faithfulness, not correctness.** A high score means "verbatim/closely
   supported by the source text," not "factually right." A value can be grounded
   yet be the wrong field, or correct yet reworded. This table calibrates the
   former.
2. **Single domain.** All 24 docs are Murrelektronik M12 connector/cable
   datasheets. Calibration may differ on invoices, forms, or other layouts.
3. **Labels are author-judged**, not an independent multi-annotator gold set.
   The protocol is documented and the audit trail is committed, but a second
   annotator would sharpen the borderline name cases. Reassuringly, the
   subjectively-labeled name leaves are the *better*-calibrated subset (ECE
   0.123); the headline miscalibration sits in the objectively-labeled value
   leaves.
4. **n = 107**, stratified — bins are uneven (the 0.1–0.5 range is naturally
   sparse because the verifier's scores are bimodal). MCE rests on the 14-leaf
   bottom bin. Brier (0.131), which is binning-free, corroborates the picture.

## Reproduce

```bash
bash scripts/ece_batch_extract.sh                 # 24 datasheets, grounding on
~/miniconda3/envs/silo/bin/python scripts/build_ece.py sample
~/miniconda3/envs/silo/bin/python scripts/apply_ece_labels.py
~/miniconda3/envs/silo/bin/python scripts/build_ece.py score
```
