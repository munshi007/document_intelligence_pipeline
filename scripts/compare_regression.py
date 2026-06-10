"""Compare a regression sweep against a baseline output dir.

Usage: python scripts/compare_regression.py <baseline_dir> <new_dir>

Per doc: non-empty leaf count, top-level list sizes, eval metrics, grounding
block (incl. coverage / recall fields). Purely read-only reporting.
"""
import json
import sys
from pathlib import Path


def leaves(o):
    if isinstance(o, dict):
        return sum(leaves(v) for v in o.values())
    if isinstance(o, list):
        return sum(leaves(v) for v in o)
    return 0 if o in (None, "", [], {}) else 1


def list_sizes(d):
    return {
        k: len(v) for k, v in d.items()
        if isinstance(v, list) and k not in ("page_references", "source_evidence")
    }


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main(base_dir: str, new_dir: str) -> None:
    base, new = Path(base_dir), Path(new_dir)
    docs = sorted(p.name for p in new.iterdir() if p.is_dir())
    for doc in docs:
        print(f"\n=== {doc} ===")
        for label, root in (("BASE", base / doc), ("NEW ", new / doc)):
            ext = load(root / f"{doc}_universal_extraction.json")
            ev = load(root / f"{doc}_eval.json")
            if not ext:
                print(f"  {label}: <missing extraction>")
                continue
            g = ext.get("_grounding") or {}
            cov = g.get("coverage") or {}
            print(
                f"  {label}: leaves={leaves(ext):4d} lists={list_sizes(ext)} "
                f"pass_rate={g.get('pass_rate')} repaired={g.get('repaired')} "
                f"flagged={g.get('flagged')}"
            )
            if cov:
                print(
                    f"        coverage: checked={cov.get('tables_checked')} "
                    f"undercovered={len(cov.get('undercovered', []))} "
                    f"recovered_items={cov.get('items_recovered')}"
                )
            if ev:
                print(
                    f"        eval: status={ev.get('status')} "
                    f"populated={ev.get('non_empty_extraction_rate')} "
                    f"g_pass={ev.get('grounding_pass_rate')} "
                    f"recall_flags={ev.get('recall_undercovered_tables', '-')} "
                    f"recall_recovered={ev.get('recall_items_recovered', '-')}"
                )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
