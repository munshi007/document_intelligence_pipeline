#!/usr/bin/env bash
# Task #54 — batch-extract the 25 gpt4o-annotated EVAL_DATA datasheets with
# grounding ON so each leaf's field_confidence lands in <stem>_grounding.json.
# Resumable: skips any doc whose grounding sidecar already exists.
set -u
cd "$(dirname "$0")/.." || exit 1

PY=~/miniconda3/envs/silo/bin/python
OUT=output/ece_run
ANN=data/ground_truth/annotations_gpt4o.jsonl
mkdir -p "$OUT"

mapfile -t PDFS < <("$PY" -c "import json,sys
[print(json.loads(l)['source_pdf']) for l in open('$ANN')]")

total=${#PDFS[@]}
i=0
for pdf in "${PDFS[@]}"; do
  i=$((i+1))
  stem=$(basename "$pdf" .pdf)
  sidecar="$OUT/${stem}_grounding.json"
  if [[ -f "$sidecar" ]]; then
    echo "[$i/$total] SKIP $stem (grounding sidecar exists)"
    continue
  fi
  echo "[$i/$total] EXTRACT $stem"
  "$PY" -m cli extract "$pdf" -o "$OUT" --with-grounding --force \
    >"$OUT/${stem}.log" 2>&1
  rc=$?
  echo "[$i/$total] done $stem rc=$rc"
done
echo "ALL DONE ($total docs)"
