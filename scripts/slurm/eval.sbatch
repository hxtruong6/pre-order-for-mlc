#!/bin/bash
#SBATCH --job-name=preorder-eval
#SBATCH --output=slurm_logs/%x_%A.out
#SBATCH --error=slurm_logs/%x_%A.err
#SBATCH --account=student
#SBATCH --qos=normal
#SBATCH --partition=DEF
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00

# Merge per-(repeat, fold) split pickles, then run evaluate.py + evaluate_ecc.py.
#
# Required env vars (set by submit_all.sh):
#   DATASET, RESULTS_DIR
# Optional:
#   NOISE_RATES (default "0.0 0.1 0.2 0.3"), BASE_LEARNER (passed through to evaluate.py if it accepts it)

set -euo pipefail

DATASET="${DATASET:?DATASET env var required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR env var required}"
NOISE_RATES="${NOISE_RATES:-0.0 0.1 0.2 0.3}"

cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research_preorder_mlc

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK}"

echo "=== host=$(hostname) job=${SLURM_JOB_ID} dataset=${DATASET} results=${RESULTS_DIR} ==="

# Merge each (dataset, noise) cell's split partials into the consolidated pickle.
for noise in ${NOISE_RATES}; do
    echo "--- merge ${DATASET} noise=${noise} ---"
    python scripts/slurm/merge_split_results.py \
        --dataset "${DATASET}" \
        --noise_rate "${noise}" \
        --results_dir "${RESULTS_DIR}" || echo "[warn] no splits for noise=${noise}"
done

echo "--- evaluate (BOPOS + CLR + BR + CC) ---"
time python scripts/evaluate.py --dataset "${DATASET}" --results_dir "${RESULTS_DIR}"

echo "--- evaluate ECC ---"
time python scripts/evaluate_ecc.py --dataset "${DATASET}" --algorithm ecc --results_dir "${RESULTS_DIR}"

echo "=== done ==="
