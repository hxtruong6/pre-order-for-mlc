#!/usr/bin/env bash
# End-to-end reproduction driver for the preorder4MLC paper.
#
# For every dataset key registered in preorder4mlc.config::ConfigManager.DATASET_CONFIGS,
# this script:
#   1. Trains the BOPOs pipeline and the CLR / BR / CC baselines (scripts/train.py).
#   2. Trains the ECC baseline (scripts/train_ecc.py).
#   3. Evaluates the BOPOs / CLR / BR / CC pickles into per-fold CSVs (scripts/evaluate.py).
#   4. Evaluates the ECC pickle (scripts/evaluate_ecc.py).
#
# Outputs land in ${RESULTS_DIR}; per-dataset stdout/stderr go to
# ${LOG_DIR}/<dataset>.log. Run preorder4mlc/utils/summarize_metrics.py afterwards
# to aggregate the CSVs into the per-dataset summary tables consumed by the
# statistical tests and figure scripts.

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results/run-$(date +%Y%m%d)}"
LOG_DIR="${LOG_DIR:-logs/run-$(date +%Y%m%d)}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

DATASETS=(
    chd_49
    emotions
    scene
    yeast
    water_quality
    gpositivepseaac
    plantpseaac
    humanpseaac
    enron
)

run_dataset() {
    local dataset="$1"
    local log_file="${LOG_DIR}/${dataset}.log"

    {
        echo "===================="
        echo "[$(date -Is)] Running ${dataset}"
        echo "===================="

        python scripts/train.py    --dataset "${dataset}" --results_dir "${RESULTS_DIR}"
        python scripts/evaluate.py --dataset "${dataset}" --results_dir "${RESULTS_DIR}"

        python scripts/train_ecc.py    --dataset "${dataset}" --algorithm ecc --results_dir "${RESULTS_DIR}"
        python scripts/evaluate_ecc.py --dataset "${dataset}" --algorithm ecc --results_dir "${RESULTS_DIR}"

        echo "[$(date -Is)] Finished ${dataset}"
    } &>"${log_file}"
}

for dataset in "${DATASETS[@]}"; do
    run_dataset "${dataset}"
done
