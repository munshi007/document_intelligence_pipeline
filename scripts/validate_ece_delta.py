"""Task #55 — re-ground the #54 sample with the CURRENT verifier and re-score ECE.

Runs ExtractorAgent._verify_string_spans over each doc's already-extracted
record against its reconstructed source, reads the new per-leaf confidence for
the 107 #54-sampled leaves, re-applies the SAME hand labels, and reports ECE.
Run it twice (git stash the agent.py change for the baseline) for a fair
old-vs-new delta on identical source text.
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor.agent import ExtractorAgent  # noqa: E402
from extractor.calibration import calibration_report, render_reliability_table  # noqa: E402
from build_ece import _load_source_text, RUN_DIR  # noqa: E402

WORKSHEET = Path("output/ece_run/label_worksheet.json")


def main():
    data = json.loads(WORKSHEET.read_text())
    pairs, miss = [], 0
    target_ids = {1, 3, 4, 7, 10, 11, 12, 13, 20, 22, 24}  # comma / +/- / scale
    target_moves = []
    for doc in data["docs"]:
        stem = doc["doc"]
        rec = json.loads((RUN_DIR / f"{stem}_universal_extraction.json").read_text())
        src = _load_source_text(stem)
        fc = ExtractorAgent._verify_string_spans(copy.deepcopy(rec), src)["field_confidence"]
        for leaf in doc["leaves"]:
            if leaf["label"] not in (0, 1):
                continue
            new = fc.get(leaf["path"])
            if new is None:
                miss += 1
                continue
            pairs.append((new, bool(leaf["label"])))
            if leaf["id"] in target_ids:
                target_moves.append((leaf["id"], leaf["conf"], new, str(leaf["value"])[:28]))

    rep = calibration_report(pairs, 10)
    print(f"re-grounded pairs={len(pairs)}  (missing={miss})")
    print(json.dumps({k: rep[k] for k in ("n", "base_rate", "mean_confidence", "ece", "mce", "brier")}, indent=2))
    print(render_reliability_table(rep))
    print("\n-- target under-confident leaves (id: worksheet_conf -> re-grounded) --")
    for _id, old, new, val in sorted(target_moves):
        print(f"  id={_id:<3} {old:.2f} -> {new:.2f}  {val!r}")


if __name__ == "__main__":
    main()
