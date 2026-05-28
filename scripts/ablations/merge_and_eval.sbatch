#!/bin/bash
#SBATCH --job-name=preorder-merge
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

DATASET="${DATASET:?DATASET env var required}"
NOISE_RATE="${NOISE_RATE:-0.0}"
RESULTS_DIR="${RESULTS_DIR:-results/full_${DATASET}}"

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

echo "=== host=$(hostname) job=${SLURM_JOB_ID} ==="
echo "=== dataset=${DATASET} noise=${NOISE_RATE} results=${RESULTS_DIR} ==="

echo
echo "--- MERGE ---"
python scripts/ablations/merge_split_results.py \
    --dataset "${DATASET}" \
    --noise_rate "${NOISE_RATE}" \
    --results_dir "${RESULTS_DIR}"

echo
echo "--- EVALUATE ---"
time python scripts/evaluate.py \
    --dataset "${DATASET}" \
    --results_dir "${RESULTS_DIR}"

echo
echo "=== done ==="
ls -la "${RESULTS_DIR}" | tail -20
