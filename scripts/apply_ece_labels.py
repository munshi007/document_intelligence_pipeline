"""Task #54 — apply the manual 'supported in source?' labels to the worksheet.

Labeling protocol (documented in docs/CALIBRATION.md):
  label 1 = the value's fact is present in that doc's source text, allowing
            case/whitespace, decimal comma<->dot, unicode/mojibake glyph noise
            (deg, ohm, +/-, squared, @), unit symbol<->spelled-out
            (m/s <-> meters per second), digits<->number-words
            (1 Mio. <-> one million / 2000000), and term reordering.
  label 0 = OCR garbage / corrupted beyond the data (non-word fragments),
            a numeric absent from the doc, or a NAME that introduces a salient
            term not in the source (amount / winding / jacket-for-cable).

Every id NOT in ZEROS is labeled 1. Rationale per zero is kept for audit.
Each judgement was made by reading the per-doc source text reconstructed from
the layout regions; see scripts/build_ece.py.
"""
import json
from pathlib import Path

WORKSHEET = Path("output/ece_run/label_worksheet.json")
LABELS_OUT = Path("output/ece_run/manual_labels.json")

# id -> reason the value is NOT supported in its source document.
ZEROS = {
    0:  "'-22': doc has only -25 C / -40 C, no -22 -> mangled, absent",
    2:  "'0  .': OCR garbage fragment",
    5:  "'⊺/m': corrupted unit glyph, not a value",
    6:  "'+Â±%5': corrupted/reordered, not a faithful value",
    8:  "'0 n': garbage fragment",
    9:  "'0 0': garbage fragment",
    14: "'I c 2 2 2 2 ...': hallucinated repetition",
    15: "'Aon 2 2 2 2 ...': hallucinated repetition",
    18: "'Filling amount': source has 'Filler', term 'amount' absent (changes meaning)",
    19: "'22.0' (min_value): no 22 in this doc (has 2 Mio., -25/-40 C)",
    23: "'u00d po': leftover escape garbage",
    26: "'Color Winding': source has 'color' but 'winding' absent",
    35: "'Jacket weight': source says 'Cable weight' (different referent)",
    36: "'kV ↵ 6la': corrupted '2,5 kV @ 60 s' -> '6la' non-term, digit lost",
}


def main():
    data = json.loads(WORKSHEET.read_text())
    audit = {}
    n1 = n0 = 0
    for doc in data["docs"]:
        for leaf in doc["leaves"]:
            lid = leaf["id"]
            if lid in ZEROS:
                leaf["label"] = 0
                n0 += 1
                audit[lid] = {"label": 0, "conf": leaf["conf"], "value": str(leaf["value"])[:60], "reason": ZEROS[lid]}
            else:
                leaf["label"] = 1
                n1 += 1
                audit[lid] = {"label": 1, "conf": leaf["conf"], "value": str(leaf["value"])[:60]}
    WORKSHEET.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    LABELS_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"applied labels: {n1} supported (1), {n0} not-supported (0), total {n1+n0}")
    print(f"wrote {LABELS_OUT}")


if __name__ == "__main__":
    main()
