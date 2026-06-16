"""Task #54 — build the grounding-confidence calibration (ECE) dataset.

Two modes:

    python scripts/build_ece.py sample   # -> output/ece_run/label_worksheet.json
    python scripts/build_ece.py score     # reads labels -> docs/CALIBRATION.md

`sample` pools every *content* leaf across the 25 EVAL_DATA datasheet extracts,
attaches each leaf's grounding confidence and its value, and draws a
confidence-stratified sample (all of the sparse low/mid bins + a capped sample
of the saturated top bin) so the labeled set spans the whole [0,1] range — the
prerequisite for a meaningful reliability curve. Each leaf is grouped under its
document together with that document's full reconstructed source text, so the
'supported in source?' judgement is made by reading, not by an automated
substring search (which would merely re-measure the verifier).

`score` reads the hand-labeled worksheet (`label` ∈ {0,1} per leaf) and emits the
calibration report + markdown table via extractor.calibration.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor.calibration import calibration_report, render_reliability_table  # noqa: E402

RUN_DIR = Path("output/ece_run")
ANN = Path("data/ground_truth/annotations_gpt4o.jsonl")
WORKSHEET = RUN_DIR / "label_worksheet.json"
REPORT_JSON = RUN_DIR / "calibration_report.json"
DOC_MD = Path("docs/CALIBRATION.md")

N_BINS = 10
PER_BIN_CAP = 30          # cap per confidence bin so the 1.0 bin can't dominate
SEED = 54                 # deterministic sampling

# Leaf paths we never calibrate: provenance metadata and structural/meta leaves,
# not extracted document content.
_SKIP_SUBSTR = ("source_evidence",)
_SKIP_LEAF = {"confidence_score", "page_number", "reasoning_thoughts", "confidence"}


def _doc_stems() -> List[str]:
    return [json.loads(l)["source_pdf"].split("/")[-1][:-4] for l in ANN.open()]


def _load_source_text(stem: str) -> str:
    """Reconstruct the per-doc source corpus the verifier grounded against:
    region text blocks + flattened table cells."""
    parts: List[str] = []
    reg = RUN_DIR / f"{stem}_regions.json"
    if reg.exists():
        for r in json.loads(reg.read_text()):
            t = r.get("text")
            if t:
                parts.append(str(t))
    man = RUN_DIR / f"{stem}_manifest.json"
    if man.exists():
        for e in json.loads(man.read_text()).get("elements", []):
            td = e.get("table_data")
            if isinstance(td, list):
                for row in td:
                    if isinstance(row, list):
                        parts.append(" | ".join(str(c) for c in row if c))
                    elif isinstance(row, dict):
                        parts.append(" | ".join(str(c) for c in row.values() if c))
    return "\n".join(parts)


_SEG = re.compile(r"([^\[.]+)(?:\[(\d+)\])?")


def _get_by_path(obj: Any, path: str) -> Any:
    """Resolve a verifier path like 'parameters[0].name' against the record."""
    cur = obj
    for seg in path.split("."):
        m = _SEG.fullmatch(seg)
        if not m or cur is None:
            return None
        key, idx = m.group(1), m.group(2)
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
        if idx is not None:
            if isinstance(cur, list) and int(idx) < len(cur):
                cur = cur[int(idx)]
            else:
                return None
    return cur


def _is_content_leaf(path: str) -> bool:
    if any(s in path for s in _SKIP_SUBSTR):
        return False
    last = path.split(".")[-1]
    last = re.sub(r"\[\d+\]", "", last)
    return last not in _SKIP_LEAF


def _collect() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stem in _doc_stems():
        gpath = RUN_DIR / f"{stem}_grounding.json"
        xpath = RUN_DIR / f"{stem}_universal_extraction.json"
        if not gpath.exists() or not xpath.exists():
            continue
        fc = json.loads(gpath.read_text()).get("field_confidence", {})
        record = json.loads(xpath.read_text())
        for path, conf in fc.items():
            if not _is_content_leaf(path):
                continue
            val = _get_by_path(record, path)
            if val is None or (isinstance(val, str) and not val.strip()):
                continue
            rows.append(
                {"doc": stem, "path": path, "value": val, "conf": float(conf)}
            )
    return rows


def _stratified(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All sparse bins kept; the saturated top bin capped to PER_BIN_CAP."""
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        idx = min(int(r["conf"] * N_BINS), N_BINS - 1)
        buckets.setdefault(idx, []).append(r)
    # Deterministic shuffle without RNG: sort by a stable (cross-process) hash.
    def _key(r):
        h = hashlib.md5(f"{SEED}|{r['doc']}|{r['path']}".encode()).hexdigest()
        return int(h[:8], 16)

    sampled: List[Dict[str, Any]] = []
    for idx in sorted(buckets):
        items = sorted(buckets[idx], key=_key)
        sampled.extend(items[:PER_BIN_CAP])
    return sampled


def do_sample() -> None:
    rows = _collect()
    sampled = _stratified(rows)
    by_doc: Dict[str, Dict[str, Any]] = {}
    for i, r in enumerate(sampled):
        d = by_doc.setdefault(
            r["doc"], {"doc": r["doc"], "source_text": _load_source_text(r["doc"]), "leaves": []}
        )
        d["leaves"].append(
            {"id": i, "path": r["path"], "value": r["value"], "conf": round(r["conf"], 3), "label": None}
        )
    out = {
        "meta": {
            "n_total_leaves": len(rows),
            "n_sampled": len(sampled),
            "n_bins": N_BINS,
            "per_bin_cap": PER_BIN_CAP,
            "label_key": "1 = value is supported in source_text, 0 = not supported",
        },
        "docs": list(by_doc.values()),
    }
    WORKSHEET.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    # Coverage print so we can see the bin spread before labeling.
    from collections import Counter
    spread = Counter(min(int(r["conf"] * N_BINS), N_BINS - 1) for r in sampled)
    print(f"pooled {len(rows)} content leaves -> sampled {len(sampled)} for labeling")
    for b in range(N_BINS):
        print(f"  bin [{b/N_BINS:.1f},{(b+1)/N_BINS:.1f}): {spread.get(b,0)}")
    print(f"wrote {WORKSHEET}")


def _labeled_pairs() -> List[Tuple[float, bool]]:
    data = json.loads(WORKSHEET.read_text())
    pairs: List[Tuple[float, bool]] = []
    unlabeled = 0
    for doc in data["docs"]:
        for leaf in doc["leaves"]:
            if leaf.get("label") in (0, 1):
                pairs.append((leaf["conf"], bool(leaf["label"])))
            else:
                unlabeled += 1
    if unlabeled:
        print(f"WARNING: {unlabeled} leaves still unlabeled (skipped)")
    return pairs


def do_score() -> None:
    pairs = _labeled_pairs()
    rep = calibration_report(pairs, n_bins=N_BINS)
    REPORT_JSON.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("n", "base_rate", "mean_confidence", "ece", "mce", "brier")}, indent=2))
    print(render_reliability_table(rep))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    if mode == "sample":
        do_sample()
    elif mode == "score":
        do_score()
    else:
        sys.exit(f"unknown mode: {mode}")
