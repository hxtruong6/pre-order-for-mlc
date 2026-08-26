#!/usr/bin/env bash
# Fast reproducibility capsule: trains + evaluates the preorder pipeline on the
# smallest dataset (CHD-49) on CPU and writes per-fold CSVs to the output dir.
# Usage: bash scripts/reproduce_capsule.sh [OUTPUT_DIR]
set -euo pipefail

OUT_DIR="${1:-results/capsule}"
DATASET="chd_49"
mkdir -p "${OUT_DIR}"

echo "[capsule] training ${DATASET} -> ${OUT_DIR}"
python scripts/train.py    --dataset "${DATASET}" --results_dir "${OUT_DIR}"
echo "[capsule] evaluating ${DATASET}"
python scripts/evaluate.py --dataset "${DATASET}" --results_dir "${OUT_DIR}"
echo "[capsule] done. CSVs under ${OUT_DIR}"
