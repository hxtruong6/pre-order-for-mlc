#!/usr/bin/env bash
# Submit 3 extra baselines (mlknn, ecc, lp) for 3 datasets = 9 jobs.
# train_extra_baselines.py loops all 4 noise levels internally, so one job
# per (dataset, algo) covers the full noise sweep.
# Respects the def-QOS MaxSubmit=40 cap shared with master_controller.
#
# Usage: nohup bash scripts/ablations/submit_extras.sh > slurm_logs/submit_extras.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/../.."

MAX_QUEUE="${MAX_QUEUE:-35}"
POLL_SECS="${POLL_SECS:-600}"

DATASETS=(
    "medical|results/full_medical_split|160G|2-00:00:00"
    "chestxray_densenet|results/full_chestxray_densenet_split|48G|1-00:00:00"
    "chestxray_resnet|results/full_chestxray_resnet_split|48G|1-00:00:00"
)
ALGOS=(mlknn ecc lp)

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

wait_for_slot() {
    while true; do
        local count
        count=$(squeue -u "$USER" -r -h -o "%j" 2>/dev/null | grep -c "^preorder-" || true)
        if [ "$count" -lt "$MAX_QUEUE" ]; then return 0; fi
        log "  [wait] preorder_count=$count >= $MAX_QUEUE, sleep ${POLL_SECS}s"
        sleep "$POLL_SECS"
    done
}

submit_one() {
    local DS=$1 ALGO=$2 RESULTS_DIR=$3 MEM=$4 TIME=$5
    # skip if all 4 noise pickles already produced
    local done_count
    done_count=$(ls "${RESULTS_DIR}"/dataset_${DS}_noisy_*_${ALGO}.pkl 2>/dev/null | wc -l)
    if [ "$done_count" -ge 4 ]; then
        log "  [skip] ${DS} ${ALGO} already done ($done_count/4 noise levels)"
        return 0
    fi
    while true; do
        wait_for_slot
        local JID
        if JID=$(sbatch --parsable \
                --time="$TIME" \
                --cpus-per-task=4 \
                --mem="$MEM" \
                --export=ALL,DATASET="$DS",RESULTS_DIR="$RESULTS_DIR",ALGO="$ALGO",BASE_LEARNER=rf \
                scripts/ablations/extra_baseline.sbatch 2>&1); then
            log "  submitted ${DS} ${ALGO} -> ${JID}"
            return 0
        fi
        log "  [retry] sbatch failed: ${JID}; sleep ${POLL_SECS}s"
        sleep "$POLL_SECS"
    done
}

log "=== START submit_extras (pid=$$, MAX_QUEUE=${MAX_QUEUE}, POLL=${POLL_SECS}s) ==="
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r DS RESULTS_DIR MEM TIME <<< "$entry"
    mkdir -p "$RESULTS_DIR"
    for ALGO in "${ALGOS[@]}"; do
        submit_one "$DS" "$ALGO" "$RESULTS_DIR" "$MEM" "$TIME"
    done
done
log "=== END submit_extras ==="
