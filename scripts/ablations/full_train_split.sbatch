#!/bin/bash
#SBATCH --job-name=preorder-split
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# Array-task body. Expects SLURM_ARRAY_TASK_ID in [0, 24].
# Maps task id → (repeat_idx, fold_idx):
#   repeat = task_id // 5
#   fold   = task_id % 5
# ALGORITHM env var must be set (bopos|clr|br|cc) — the submission
# controller submits four arrays per (dataset, noise) cell, one per
# algorithm. This keeps each array at 25 tasks so it fits inside the
# def-QOS MaxSubmitJobs=40 cap.

set -euo pipefail

DATASET="${DATASET:?DATASET env var required}"
NOISE_RATE="${NOISE_RATE:?NOISE_RATE env var required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR env var required}"
ALGORITHM="${ALGORITHM:?ALGORITHM env var required (bopos|clr|br|cc)}"
BASE_LEARNER="${BASE_LEARNER:-RF}"
SOLVER="${SOLVER:-highs}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?must be run as an array job}"
REPEAT=$(( TASK_ID / 5 ))
FOLD=$(( TASK_ID % 5 ))

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

echo "=== host=$(hostname) job=${SLURM_JOB_ID} task=${TASK_ID} cpus=${SLURM_CPUS_PER_TASK:-?} ==="
echo "=== dataset=${DATASET} noise=${NOISE_RATE} results=${RESULTS_DIR} ==="
echo "=== algorithm=${ALGORITHM} repeat=${REPEAT} fold=${FOLD} solver=${SOLVER} base=${BASE_LEARNER} ==="

mkdir -p "${RESULTS_DIR}"

N_JOBS="${N_JOBS:-4}"
export LOKY_MAX_CPU_COUNT="${N_JOBS}"
export JOBLIB_MAX_NB_BYTES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
EFFECTIVE_CPUS=$(python -c "import os; print(len(os.sched_getaffinity(0)))")
CAP=$(( EFFECTIVE_CPUS < N_JOBS ? EFFECTIVE_CPUS : N_JOBS ))
TASKSET_PREFIX=""
if [ "${EFFECTIVE_CPUS}" -gt "${CAP}" ]; then
    CPU_LIST=$(python -c "import os; print(','.join(map(str, sorted(os.sched_getaffinity(0))[:${CAP}])))")
    TASKSET_PREFIX="taskset -c ${CPU_LIST}"
fi
echo "=== cgroup_cpus=${EFFECTIVE_CPUS} cap=${CAP} prefix='${TASKSET_PREFIX}' ==="

echo
echo "--- TRAIN (split) ---"
time ${TASKSET_PREFIX} python scripts/train.py \
    --dataset "${DATASET}" \
    --noise_rate "${NOISE_RATE}" \
    --results_dir "${RESULTS_DIR}" \
    --base_learner "${BASE_LEARNER}" \
    --solver "${SOLVER}" \
    --algorithm "${ALGORITHM}" \
    --repeat_idx "${REPEAT}" \
    --fold_idx "${FOLD}"

echo
echo "=== done ==="
