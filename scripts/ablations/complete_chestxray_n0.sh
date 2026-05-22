#!/usr/bin/env bash
# Submit the remaining work to fully cover chestxray_densenet noise=0.0:
#   - cc       full 25 (R,F): --array=0-24
#   - bopos    missing 17:     --array=8-24
#   - clr      missing 17:     --array=8-24
#   - br       missing 17:     --array=8-24
# Total 76 tasks; cap 40 = 15 GPU + 25 free, so stagger:
#   1) submit cc (25)
#   2) when preorder count <= 8, submit next (17 fits inside 25 free)
#   3) repeat for bopos / clr / br
# After all 4 arrays submitted, sbatch a merge+eval job depending on all of them.

set -uo pipefail
cd "$(dirname "$0")/../.."

POLL_SECS="${POLL_SECS:-1800}"   # 30 min
DATASET=chestxray_densenet
NOISE=0.0
RESULTS_DIR=results/full_chestxray_densenet_split
LOG=slurm_logs/complete_chestxray_n0.log

mkdir -p "$RESULTS_DIR"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

preorder_count() {
    squeue -u "$USER" -r -h -o "%j" 2>/dev/null | grep -c "^preorder-" || true
}

wait_until_below() {
    local THRESH=$1
    while true; do
        local n
        n=$(preorder_count)
        if [ "$n" -le "$THRESH" ]; then return 0; fi
        log "  [wait] preorder_tasks=$n > $THRESH, sleeping ${POLL_SECS}s"
        sleep "$POLL_SECS"
    done
}

submit_array() {
    local ALGO=$1 RANGE=$2 TASKS=$3
    # Need (15 GPU + TASKS) <= 40, i.e. TASKS <= 25 - any preorder already
    # in queue. We wait until preorder_count <= 25 - TASKS.
    local THRESH=$(( 25 - TASKS ))
    wait_until_below "$THRESH"
    while true; do
        local JID
        if JID=$(sbatch --parsable \
                --array="$RANGE" \
                --time=1-00:00:00 \
                --cpus-per-task=4 \
                --mem=48G \
                --export=ALL,DATASET="$DATASET",NOISE_RATE="$NOISE",RESULTS_DIR="$RESULTS_DIR",ALGORITHM="$ALGO",BASE_LEARNER=RF,SOLVER=highs \
                scripts/ablations/full_train_split.sbatch 2>&1); then
            log "submitted $ALGO array=$RANGE ($TASKS tasks) → $JID"
            echo "$JID" >> /tmp/complete_chestxray_n0_jids.txt
            return 0
        fi
        log "  [retry] sbatch failed: $JID; sleeping ${POLL_SECS}s"
        sleep "$POLL_SECS"
    done
}

rm -f /tmp/complete_chestxray_n0_jids.txt
log "=== START completion of chestxray_densenet noise=0.0 (pid=$$, poll=${POLL_SECS}s) ==="

submit_array cc    "0-24" 25
submit_array bopos "8-24" 17
submit_array clr   "8-24" 17
submit_array br    "8-24" 17

# Build dependency list from JIDs
DEPS=""
while read -r jid; do DEPS="${DEPS}:${jid}"; done < /tmp/complete_chestxray_n0_jids.txt
DEPS="${DEPS#:}"
log "all 4 arrays submitted: $DEPS"

# Submit merge+evaluate after all 4 finish (also depends on the original 3
# already-completed arrays 127626/127627/127628 — but those are done, so afterany
# is satisfied immediately; we still list them as a safety belt).
while true; do
    if MJID=$(sbatch --parsable \
            --dependency="afterany:${DEPS}" \
            --export=ALL,DATASET="$DATASET",NOISE_RATE="$NOISE",RESULTS_DIR="$RESULTS_DIR" \
            scripts/ablations/merge_and_eval.sbatch 2>&1); then
        log "merge+eval job: $MJID (deps=afterany:${DEPS})"
        break
    fi
    log "  [retry] merge sbatch failed: $MJID; sleeping ${POLL_SECS}s"
    sleep "$POLL_SECS"
done

log "=== END all jobs queued ==="
