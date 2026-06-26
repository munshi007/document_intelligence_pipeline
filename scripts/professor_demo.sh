#!/usr/bin/env bash
# =============================================================================
# professor_demo.sh — how to run the feat/schema-hints branch from scratch.
#
# This is a READ-ALONG reference, not a one-shot. Run the blocks you need.
# Each block is self-contained and commented. Pick A (full, needs GPU) or
# B (CPU-only, no GPU) for the live demo.
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# 0. Prerequisites (professor's machine)
# -----------------------------------------------------------------------------
#   - OS:    Linux (Ubuntu 22.04 recommended)
#   - GPU:   NVIDIA, ~24 GB VRAM (RTX 3090 class) + CUDA 12.4 host driver
#            -> required ONLY for the real pipeline (block A). Block B (tests
#               + committed calibration artifacts) runs on CPU, no GPU.
#   - Disk:  ~45 GB free at $HF_HOME (~/.cache/huggingface) for the model cache
#   - Conda: Miniconda/Anaconda on PATH
# -----------------------------------------------------------------------------

REPO="https://github.com/munshi007/document_intelligence_pipeline.git"
BRANCH="feat/schema-hints"
PY="$HOME/miniconda3/envs/silo/bin/python"   # python inside the 'silo' env

# -----------------------------------------------------------------------------
# 1. ONE-TIME SETUP  (clone -> env -> checkout branch -> download models)
# -----------------------------------------------------------------------------
git clone "$REPO"
cd document_intelligence_pipeline
git checkout "$BRANCH"                 # <-- the calibration + hint-fields work

conda env create -f environment.yml    # creates the isolated env named "silo"
conda activate silo

# Pinned-model download + smoke test (~38 GB, asks to confirm; --yes to skip).
# Exits 0 and writes BOOTSTRAP_OK if the repo is reproducibly installed.
python scripts/bootstrap.py
# Air-gapped after first run: export HF_HUB_OFFLINE=1

# -----------------------------------------------------------------------------
# 2. DEMO  A  —  FULL PIPELINE  (needs GPU)
#    Auto-discovers the schema, extracts structured JSON, AND grounds every
#    value back to the source text (per-leaf field_confidence sidecar).
# -----------------------------------------------------------------------------
python -m cli extract data/PDFS/8484.pdf \
    -o output/demo \
    --with-grounding \
    --force
# Outputs in output/demo/:
#   8484_universal_extraction.json   <- the structured extraction (main output)
#   8484_grounding.json              <- per-leaf field_confidence in [0,1]
#   8484_auto_schema.json            <- the schema the system synthesised
#   8484_discovery.json              <- domain routing + confidence

# 2b. The branch's namesake feature — FORCE specific fields into the schema
#     (populated if present in the doc, null if genuinely absent):
python -m cli extract data/PDFS/8484.pdf \
    -o output/demo_hint \
    --hint-fields "part_number,operating_temperature,protection_class" \
    --force

# -----------------------------------------------------------------------------
# 3. DEMO  B  —  CPU-ONLY  (no GPU; good if the meeting room has no GPU)
#    Shows the novel calibration result without running any model.
# -----------------------------------------------------------------------------
# 3a. The committed calibration result (Killer #2 / task #54):
cat docs/CALIBRATION.md                       # full writeup + reliability diagram
cat docs/calibration_data/calibration_report.json   # the scored numbers
# Headline: ECE 0.165, MCE 0.571, Brier 0.131 over 107 hand-labelled leaves.

# 3b. Run the calibration + grounding-normalization unit tests (fast, CPU):
python - <<'PYEOF'
import importlib, inspect, sys
sys.path.insert(0, ".")
mods = [
    "tests.extraction.test_calibration_metrics",      # ECE/MCE/Brier math
    "tests.extraction.test_grounding_normalization",  # #55 notation folding
]
passed = failed = 0
for mname in mods:
    m = importlib.import_module(mname)
    for name, fn in sorted(vars(m).items()):
        if name.startswith("test_") and inspect.isfunction(fn) and inspect.getmodule(fn) is m:
            try:
                fn(); passed += 1
            except Exception as e:
                failed += 1; print(f"FAIL {mname}.{name}: {e}")
print(f"\n{passed} passed, {failed} failed")
PYEOF

# -----------------------------------------------------------------------------
# 4. REPRODUCE the calibration table end-to-end (needs GPU + the 24 datasheets)
# -----------------------------------------------------------------------------
# bash scripts/ece_batch_extract.sh        # 24 datasheets, grounding on
# $PY scripts/build_ece.py sample          # stratified 107-leaf sample
# $PY scripts/apply_ece_labels.py          # apply the hand labels
# $PY scripts/build_ece.py score           # -> ECE / MCE / Brier + diagram
