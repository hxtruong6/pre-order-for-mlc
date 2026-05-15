#!/usr/bin/env bash
# End-to-end reproduction driver for the preorder4MLC paper.
#
# For every dataset key registered in config.py::ConfigManager.DATASET_CONFIGS,
# this script:
#   1. Trains the BOPOs pipeline and the CLR / BR / CC baselines (main.py).
#   2. Trains the MLkNN / ECC / LP baselines (extra_baselines.py).
#   3. Evaluates the BOPOs pickles into per-fold CSVs (evaluation_test.py).
#   4. Evaluates the extra-baseline pickles (evaluate_extra_baselines.py).
#
# Outputs land in ${RESULTS_DIR}; per-dataset stdout/stderr go to
# ${LOG_DIR}/<dataset>.log. Run utils/summarize_metrics.py afterwards to
# aggregate the CSVs into the per-dataset summary tables consumed by the
# statistical tests and figure scripts.

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results/20260514_v2}"
LOG_DIR="${LOG_DIR:-logs/20260514_v2}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

DATASETS=(
    chd_49
    emotions
    viruspseaac
    gpositivepseaac
    plantpseaac
    water_quality
    scene
    yeast
    humanpseaac
)

EXTRA_BASELINES=(mlknn ecc lp)

run_dataset() {
    local dataset="$1"
    local log_file="${LOG_DIR}/${dataset}.log"

    {
        echo "===================="
        echo "[$(date -Is)] Running ${dataset}"
        echo "===================="

        python main.py --dataset "${dataset}" --results_dir "${RESULTS_DIR}"
        python evaluation_test.py --dataset "${dataset}" --results_dir "${RESULTS_DIR}"

        for algo in "${EXTRA_BASELINES[@]}"; do
            python extra_baselines.py \
                --dataset "${dataset}" \
                --algorithm "${algo}" \
                --results_dir "${RESULTS_DIR}"
            python evaluate_extra_baselines.py \
                --dataset "${dataset}" \
                --algorithm "${algo}" \
                --results_dir "${RESULTS_DIR}"
        done

        echo "[$(date -Is)] Finished ${dataset}"
    } &>"${log_file}"
}

for dataset in "${DATASETS[@]}"; do
    run_dataset "${dataset}"
done
