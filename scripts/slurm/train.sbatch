#!/bin/bash
#SBATCH --job-name=preorder-train
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --account=student
#SBATCH --qos=normal
#SBATCH --partition=DEF
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# Array task body. Expects SLURM_ARRAY_TASK_ID in [0, 24] for RF (5 reps × 5 folds)
# or [0, 4] for LGBM (1 rep × 5 folds — controller submits with --array=0-4).
# Maps task id → (repeat_idx, fold_idx):
#   repeat = task_id / 5
#   fold   = task_id % 5
#
# Required env vars (set by submit_all.sh via sbatch --export):
#   DATASET, NOISE_RATE, RESULTS_DIR, ALGORITHM (bopos|clr|br|cc|ecc), BASE_LEARNER (RF|LightGBM)

set -euo pipefail

DATASET="${DATASET:?DATASET env var required}"
NOISE_RATE="${NOISE_RATE:?NOISE_RATE env var required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR env var required}"
ALGORITHM="${ALGORITHM:?ALGORITHM env var required}"
BASE_LEARNER="${BASE_LEARNER:-RF}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?must be run as an array job}"
REPEAT=$(( TASK_ID / 5 ))
FOLD=$(( TASK_ID % 5 ))

cd "${SLURM_SUBMIT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research_preorder_mlc

echo "=== host=$(hostname) job=${SLURM_JOB_ID} task=${TASK_ID} cpus=${SLURM_CPUS_PER_TASK} ==="
echo "=== dataset=${DATASET} noise=${NOISE_RATE} algo=${ALGORITHM} learner=${BASE_LEARNER} ==="
echo "=== repeat=${REPEAT} fold=${FOLD} results=${RESULTS_DIR} ==="

mkdir -p "${RESULTS_DIR}"

# Cap thread oversubscription: joblib n_jobs inside the prediction step would
# otherwise multiply with BLAS threads and starve the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Cores actually available to this task. Normally SLURM_CPUS_PER_TASK (=4 here),
# but some array submissions land with SLURM_CPUS_PER_TASK=1 even when more cores
# are allocated on the node (observed on job 154949: 16 cores allocated yet
# cpus-per-task=1, so the ILP ran single-threaded and was heading for a 48h
# TIMEOUT). Fall back to the larger node allocation so the per-instance ILP
# always uses every core it was actually given.
NCORES="${SLURM_CPUS_PER_TASK:-1}"
if [ "${SLURM_CPUS_ON_NODE:-1}" -gt "${NCORES}" ]; then
    NCORES="${SLURM_CPUS_ON_NODE}"
fi
echo "=== NCORES=${NCORES} (cpus_per_task=${SLURM_CPUS_PER_TASK:-?} cpus_on_node=${SLURM_CPUS_ON_NODE:-?}) ==="
export LOKY_MAX_CPU_COUNT="${NCORES}"
# Parallelize the per-instance ILP search across NCORES workers. Default in
# searching_algorithms.py is 1 (sequential); without this enron (K=53) takes
# ~17h per ILP combo and TIMEOUTs the 48h walltime.
SEARCH_N_JOBS="${NCORES}"
# enron (K=53) BOPOS parallelism. Each per-instance GLPK solve factorizes a
# ~140k-row LP basis taking ~tens of seconds, so a fold is factorization-bound.
# Measured ceiling is only ~4.5x effective regardless of n_jobs/batch_size — the
# workload rarely has many slow instances in flight at once (NOT a CPU-binding
# limit: cores are unconfined, the node grants all SLURM_CPUS_ON_NODE). Peak
# memory at 10-way is ~224 GiB (job 165956), well under the 360G QOS cap. n_jobs
# does NOT change the ILP solution (results bit-identical), only memory; 10 is a
# safe value that leaves headroom for the heavier high-noise folds.
ENRON_SEARCH_N_JOBS="${ENRON_SEARCH_N_JOBS:-10}"
if [ "${DATASET}" = "enron" ] && [ "${ALGORITHM}" = "bopos" ] && [ "${SEARCH_N_JOBS}" -gt "${ENRON_SEARCH_N_JOBS}" ]; then
    SEARCH_N_JOBS="${ENRON_SEARCH_N_JOBS}"
    echo "=== enron BOPOS: capping search parallelism ${NCORES} -> ${ENRON_SEARCH_N_JOBS} ==="
fi
export PREORDER_SEARCH_N_JOBS="${SEARCH_N_JOBS}"

# Solver choice. The paper baseline is GLPK (cvxopt.glpk.ilp) and every dataset
# uses it so the published results / canonical smoke hash stay bit-identical.
# enron (K=53) is the sole exception: its per-instance ILP has ~140k transitivity
# constraints and ~5.5k binary vars, where GLPK needs ~150k simplex iterations and
# does not finish within the 48h walltime. HiGHS (scipy.optimize.milp) solves the
# SAME standard-form MILP and reaches the SAME optimal objective (verified: max
# objective diff ~1e-8 across all smoke configs); only tie-breaks among alternate
# optima differ. So enron-BOPOS uses HiGHS purely as a faster exact solver.
if [ "${DATASET}" = "enron" ] && [ "${ALGORITHM}" = "bopos" ]; then
    # Solver override-able from the caller's env for A/B testing; defaults to glpk.
    # ROOT CAUSE of the OOMs (diagnosed 2026-05-31): each per-instance ILP has
    # ~140k transitivity constraints, so the LP *relaxation* basis factorization
    # alone is tens-to-hundreds of GiB — NOT the B&B tree. node_limit/mip_rel_gap
    # therefore never helped (240G OOM regardless). The only thing that bounds a
    # single hard solve is a per-instance TIME cap that returns the incumbent.
    export PREORDER_SOLVER="${PREORDER_SOLVER:-glpk}"
    if [ "${PREORDER_SOLVER}" = "highs" ]; then
        export HIGHS_MIP_REL_GAP="${HIGHS_MIP_REL_GAP:-1e-3}"
        export HIGHS_NODE_LIMIT="${HIGHS_NODE_LIMIT:-2000}"
        export HIGHS_TIME_LIMIT="${HIGHS_TIME_LIMIT:-300}"
        echo "=== solver=highs gap=${HIGHS_MIP_REL_GAP} nodes=${HIGHS_NODE_LIMIT} tlim=${HIGHS_TIME_LIMIT}s (enron BOPOS) ==="
    else
        # GLPK per-instance MIP time cap: returns the best incumbent found so far
        # (an integer-feasible preorder is found in ms), so memory stays bounded.
        # Only the hard noisy-instance minority ever hits this; the easy majority
        # solve to optimum in <1s, so a tighter cap mainly buys throughput.
        export GLPK_TIME_LIMIT="${GLPK_TIME_LIMIT:-60}"
        echo "=== solver=glpk per-instance tlim=${GLPK_TIME_LIMIT}s (enron BOPOS; K=53) ==="
    fi
else
    export PREORDER_SOLVER=glpk
fi

echo "--- TRAIN ---"
if [ "${ALGORITHM}" = "ecc" ]; then
    # train_ecc.py uses different CLI: lowercase base_learner, --repeat/--fold,
    # --noisy_rate (with --noise_rate alias).
    BL_LOWER=$(echo "${BASE_LEARNER}" | tr '[:upper:]' '[:lower:]')
    [ "${BL_LOWER}" = "lightgbm" ] && BL_LOWER="lgbm"
    time python scripts/train_ecc.py \
        --dataset "${DATASET}" \
        --noise_rate "${NOISE_RATE}" \
        --results_dir "${RESULTS_DIR}" \
        --base_learner "${BL_LOWER}" \
        --algorithm ecc \
        --repeat "${REPEAT}" \
        --fold "${FOLD}"
else
    time python scripts/train.py \
        --dataset "${DATASET}" \
        --noise_rate "${NOISE_RATE}" \
        --results_dir "${RESULTS_DIR}" \
        --base_learner "${BASE_LEARNER}" \
        --algorithm "${ALGORITHM}" \
        --repeat_idx "${REPEAT}" \
        --fold_idx "${FOLD}"
fi

echo "=== done ==="
