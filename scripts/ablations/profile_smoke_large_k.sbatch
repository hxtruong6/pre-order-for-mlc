#!/bin/bash
#SBATCH --job-name=preorder-prof
#SBATCH --partition=SMALL
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

DATASET="${DATASET:-enron}"
BASE_LEARNER="${BASE_LEARNER:-lgbm}"
TRAIN_FRAC="${TRAIN_FRAC:-0.5}"
ORDER="${ORDER:-PR}"
OUT_DIR="${OUT_DIR:-results/profile_large_k}"

mkdir -p "${OUT_DIR}"
PROF="${OUT_DIR}/${DATASET}_${BASE_LEARNER}_tf${TRAIN_FRAC//./}_${SLURM_JOB_ID}.prof"

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

echo "=== host=$(hostname)  job=${SLURM_JOB_ID}  cpus=${SLURM_CPUS_PER_TASK} ==="
echo "=== dataset=${DATASET}  base=${BASE_LEARNER}  train_frac=${TRAIN_FRAC}  order=${ORDER} ==="
echo "=== prof=${PROF} ==="
echo

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python -m cProfile -o "${PROF}" \
    scripts/ablations/smoke_large_k.py \
        --dataset "${DATASET}" \
        --order "${ORDER}" \
        --train_frac "${TRAIN_FRAC}" \
        --base_learner "${BASE_LEARNER}"

echo
echo "=== Top 30 cumulative ==="
python -c "
import pstats
st = pstats.Stats('${PROF}').sort_stats('cumulative')
st.print_stats(30)
"

echo
echo "=== Top 30 tottime ==="
python -c "
import pstats
st = pstats.Stats('${PROF}').sort_stats('tottime')
st.print_stats(30)
"

echo "=== done: ${PROF} ==="
