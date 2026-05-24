#!/bin/bash
#SBATCH --job-name=preorder-extra
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

DATASET="${DATASET:?DATASET env var required}"
ALGO="${ALGO:?ALGO env var required: mlknn|ecc|lp}"
BASE_LEARNER="${BASE_LEARNER:-rf}"
RESULTS_DIR="${RESULTS_DIR:-results/full_${DATASET}}"

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

echo "=== host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK} ==="
echo "=== dataset=${DATASET} algo=${ALGO} base=${BASE_LEARNER} results=${RESULTS_DIR} ==="

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
NOISE_RATE_ARG=""
if [ -n "${NOISE_RATE:-}" ]; then
    NOISE_RATE_ARG="--noise_rate ${NOISE_RATE}"
    echo "=== noise_rate=${NOISE_RATE} ==="
fi
NO_IS_UNBALANCE_ARG=""
if [ "${NO_IS_UNBALANCE:-0}" = "1" ]; then
    NO_IS_UNBALANCE_ARG="--no_is_unbalance"
    echo "=== no_is_unbalance=1 ==="
fi
time ${TASKSET_PREFIX} python scripts/train_extra_baselines.py \
    --dataset "${DATASET}" \
    --results_dir "${RESULTS_DIR}" \
    --algorithm "${ALGO}" \
    --base_learner "${BASE_LEARNER}" \
    ${NOISE_RATE_ARG} \
    ${NO_IS_UNBALANCE_ARG}

echo "=== done ==="
ls -la "${RESULTS_DIR}" | grep -i "${ALGO}" || true
