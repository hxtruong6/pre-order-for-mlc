#!/bin/bash
#SBATCH --job-name=preorder-train
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

DATASET="${DATASET:?DATASET env var required}"
NOISE_RATE="${NOISE_RATE:-0.0}"
BASE_LEARNER="${BASE_LEARNER:-}"
SOLVER="${SOLVER:-}"
RESULTS_DIR="${RESULTS_DIR:-results/full_${DATASET}}"
ALGORITHM="${ALGORITHM:-}"
REPEAT_IDX="${REPEAT_IDX:-}"
FOLD_IDX="${FOLD_IDX:-}"

BL_FLAG=""
if [ -n "${BASE_LEARNER}" ]; then
    BL_FLAG="--base_learner ${BASE_LEARNER}"
fi
SOLVER_FLAG=""
if [ -n "${SOLVER}" ]; then
    SOLVER_FLAG="--solver ${SOLVER}"
fi
ALGO_FLAG=""
if [ -n "${ALGORITHM}" ]; then
    ALGO_FLAG="--algorithm ${ALGORITHM}"
fi
REPEAT_FLAG=""
if [ -n "${REPEAT_IDX}" ]; then
    REPEAT_FLAG="--repeat_idx ${REPEAT_IDX}"
fi
FOLD_FLAG=""
if [ -n "${FOLD_IDX}" ]; then
    FOLD_FLAG="--fold_idx ${FOLD_IDX}"
fi

# A run is "split" if any of algorithm / repeat_idx / fold_idx is set.
# Split jobs only run train.py; the merge + evaluate is done in a separate
# job that depends on all splits for a given (dataset, noise) cell.
IS_SPLIT=0
if [ -n "${ALGORITHM}" ] || [ -n "${REPEAT_IDX}" ] || [ -n "${FOLD_IDX}" ]; then
    IS_SPLIT=1
fi

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

echo "=== host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK} ==="
echo "=== dataset=${DATASET} noise=${NOISE_RATE} results=${RESULTS_DIR} ==="
echo "=== split=${IS_SPLIT} algorithm='${ALGORITHM}' repeat_idx='${REPEAT_IDX}' fold_idx='${FOLD_IDX}' ==="

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
echo "--- TRAIN ---"
time ${TASKSET_PREFIX} python scripts/train.py \
    --dataset "${DATASET}" \
    --noise_rate "${NOISE_RATE}" \
    --results_dir "${RESULTS_DIR}" \
    ${BL_FLAG} ${SOLVER_FLAG} ${ALGO_FLAG} ${REPEAT_FLAG} ${FOLD_FLAG}

if [ "${IS_SPLIT}" = "0" ]; then
    echo
    echo "--- EVALUATE ---"
    time ${TASKSET_PREFIX} python scripts/evaluate.py \
        --dataset "${DATASET}" \
        --results_dir "${RESULTS_DIR}"
fi

echo
echo "=== done ==="
ls -la "${RESULTS_DIR}" | tail -20
