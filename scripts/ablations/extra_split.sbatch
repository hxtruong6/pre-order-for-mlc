#!/bin/bash
#SBATCH --job-name=preorder-extra-split
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

DATASET="${DATASET:?DATASET env var required}"
ALGO="${ALGO:?ALGO env var required: mlknn|ecc|lp}"
NOISE_RATE="${NOISE_RATE:?NOISE_RATE env var required}"
BASE_LEARNER="${BASE_LEARNER:-rf}"
RESULTS_DIR="${RESULTS_DIR:-results/full_${DATASET}_split}"

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

TID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required (0-24)}"
REPEAT=$(( TID / 5 ))
FOLD=$(( TID % 5 ))

echo "=== host=$(hostname) job=${SLURM_ARRAY_JOB_ID}.${TID} cpus=${SLURM_CPUS_PER_TASK} ==="
echo "=== dataset=${DATASET} algo=${ALGO} noise=${NOISE_RATE} R=${REPEAT} F=${FOLD} ==="

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

time ${TASKSET_PREFIX} python scripts/train_extra_baselines.py \
    --dataset "${DATASET}" \
    --results_dir "${RESULTS_DIR}" \
    --algorithm "${ALGO}" \
    --base_learner "${BASE_LEARNER}" \
    --noisy_rate "${NOISE_RATE}" \
    --repeat "${REPEAT}" \
    --fold "${FOLD}"

echo "=== done ==="
