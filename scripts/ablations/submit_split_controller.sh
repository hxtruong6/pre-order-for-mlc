#!/usr/bin/env bash
# Controller that submits the full split matrix in waves, respecting the
# def-QOS MaxSubmitJobs=40 cap per user.
#
# Matrix: 3 datasets × 4 noise levels × 4 algorithms = 48 array submissions,
# each with 25 tasks (5 repeats × 5 folds) → 1200 SLURM jobs. Plus 12
# merge+evaluate jobs, one per (dataset, noise), with afterany dependency
# on the four algorithm arrays for that cell.
#
# Each array is one (dataset, noise, algorithm) tuple. The controller polls
# squeue and submits a new array when the user has spare capacity.
#
# Usage:
#   nohup bash scripts/ablations/submit_split_controller.sh \\
#       > slurm_logs/submit_controller.log 2>&1 &
# Stop with `kill <pid>` (in-flight submissions are NOT cancelled).

set -uo pipefail   # not -e: we handle sbatch errors with retry

cd "$(dirname "$0")/../.."

# Each submission is a 25-task array. SLURM's MaxSubmitJobs=40 per user,
# so we must keep queue ≤ 40 - 25 = 15 before submitting another array.
# User asked for threshold "queue < 20"; we tighten to 15 to respect the
# +25 delta of each submission.
MAX_QUEUE="${MAX_QUEUE:-15}"
POLL_SECS="${POLL_SECS:-1800}"  # poll every 30 minutes
SPLIT_SBATCH="scripts/ablations/full_train_split.sbatch"
MERGE_SBATCH="scripts/ablations/merge_and_eval.sbatch"

# (dataset|results_dir|mem|time) tuples.
DATASETS=(
    "medical|results/full_medical_split|96G|2-00:00:00"
    "chestxray_densenet|results/full_chestxray_densenet_split|48G|1-00:00:00"
    "chestxray_resnet|results/full_chestxray_resnet_split|48G|1-00:00:00"
)
NOISES=(0.0 0.1 0.2 0.3)
ALGORITHMS=(bopos clr br cc)

declare -A CELL_DEPS   # key="DATASET|NOISE|RESULTS_DIR" -> ":jid1:jid2:..."

wait_for_slot() {
    # Count only def-QOS preorder array tasks (-r expands array jobs into one
    # row per task; GPU LLM jobs are in separate QOS and do not consume the
    # def-QOS submit cap of 40).
    while true; do
        local count
        count=$(squeue -u "$USER" -r -h -o "%j" 2>/dev/null | grep -c "^preorder-" || true)
        if [ "$count" -lt "$MAX_QUEUE" ]; then
            return 0
        fi
        echo "  [wait] preorder_tasks=$count >= ${MAX_QUEUE}, sleeping ${POLL_SECS}s …"
        sleep "$POLL_SECS"
    done
}

submit_array() {
    local DATASET="$1" NOISE="$2" ALGO="$3" RESULTS_DIR="$4" MEM="$5" TIME="$6"
    while true; do
        wait_for_slot
        local JID
        if JID=$(sbatch --parsable \
                --array=0-24 \
                --time="$TIME" \
                --cpus-per-task=4 \
                --mem="$MEM" \
                --export=ALL,DATASET="$DATASET",NOISE_RATE="$NOISE",RESULTS_DIR="$RESULTS_DIR",ALGORITHM="$ALGO",BASE_LEARNER=RF,SOLVER=highs \
                "$SPLIT_SBATCH" 2>&1); then
            echo "  submitted ${DATASET} noise=${NOISE} algo=${ALGO} → array job ${JID}"
            local key="${DATASET}|${NOISE}|${RESULTS_DIR}"
            CELL_DEPS["$key"]="${CELL_DEPS[$key]:-}:${JID}"
            return 0
        fi
        echo "  [retry] sbatch failed: ${JID}; sleeping ${POLL_SECS}s and retrying"
        sleep "$POLL_SECS"
    done
}

submit_merges() {
    echo
    echo "=== queueing merge+eval jobs (one per cell, depends on its 4 algo arrays) ==="
    for key in "${!CELL_DEPS[@]}"; do
        IFS='|' read -r DATASET NOISE RESULTS_DIR <<< "$key"
        local deps="${CELL_DEPS[$key]}"
        deps="${deps#:}"
        while true; do
            wait_for_slot
            local MJID
            if MJID=$(sbatch --parsable \
                    --dependency="afterany:${deps//:/:}" \
                    --export=ALL,DATASET="$DATASET",NOISE_RATE="$NOISE",RESULTS_DIR="$RESULTS_DIR" \
                    "$MERGE_SBATCH" 2>&1); then
                echo "  merge ${DATASET} noise=${NOISE} → job ${MJID} (deps=${deps})"
                break
            fi
            echo "  [retry] merge sbatch failed: ${MJID}; sleeping ${POLL_SECS}s"
            sleep "$POLL_SECS"
        done
    done
}

echo "=== controller start $(date) ==="
echo "MAX_QUEUE=${MAX_QUEUE} POLL_SECS=${POLL_SECS}"

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r DATASET RESULTS_DIR MEM TIME <<< "$entry"
    mkdir -p "$RESULTS_DIR"
    for NOISE in "${NOISES[@]}"; do
        # Skip the (chestxray_densenet, 0.0) cell — already covered by
        # complete_chestxray_n0.sh running in parallel.
        if [ "$DATASET" = "chestxray_densenet" ] && [ "$NOISE" = "0.0" ]; then
            echo "  [skip] $DATASET noise=$NOISE handled by complete_chestxray_n0.sh"
            continue
        fi
        for ALGO in "${ALGORITHMS[@]}"; do
            submit_array "$DATASET" "$NOISE" "$ALGO" "$RESULTS_DIR" "$MEM" "$TIME"
        done
    done
done

submit_merges

echo
echo "=== controller done $(date) ==="
