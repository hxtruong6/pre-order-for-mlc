#!/usr/bin/env bash
# Lightweight overnight monitor. Polls every POLL_SECS, appends a timestamped
# status block to slurm_logs/overnight_monitor.log. Records:
#   - controller process status
#   - preorder task counts (PD / R / CG / total)
#   - any state transitions of the 3 tracked array jobs
#   - any new evaluation_*.csv / merged pickle files
# Exits when STOP_FILE exists or when every tracked array has finished.
#
# Usage:
#   nohup bash scripts/ablations/overnight_monitor.sh \\
#       > slurm_logs/overnight_monitor.out 2>&1 &
# Stop: touch /tmp/overnight_monitor.stop

set -uo pipefail
cd "$(dirname "$0")/../.."

POLL_SECS="${POLL_SECS:-600}"   # 10 min
LOG="slurm_logs/overnight_monitor.log"
STOP_FILE="/tmp/overnight_monitor.stop"

TRACKED_JOBS=(127626 127627 127628)
RESULTS_DIRS=(
    results/full_chestxray_densenet_split
    results/full_chestxray_resnet_split
    results/full_medical_split
)

declare -A LAST_STATE
PREV_PARTIALS=0

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }

log "=== overnight monitor START (pid=$$, poll=${POLL_SECS}s) ==="
log "tracked arrays: ${TRACKED_JOBS[*]}"

while true; do
    if [ -f "$STOP_FILE" ]; then
        log "stop file detected ($STOP_FILE), exiting"
        rm -f "$STOP_FILE"
        break
    fi

    # task counts
    PD=$(squeue -u "$USER" -r -h -t PD -o "%j" 2>/dev/null | grep -c "^preorder-" || true)
    RR=$(squeue -u "$USER" -r -h -t R -o "%j" 2>/dev/null | grep -c "^preorder-" || true)
    CG=$(squeue -u "$USER" -r -h -t CG -o "%j" 2>/dev/null | grep -c "^preorder-" || true)
    TOTAL=$((PD + RR + CG))
    log "queue: PD=$PD R=$RR CG=$CG total=$TOTAL"

    # tracked array transitions
    still_alive=0
    for j in "${TRACKED_JOBS[@]}"; do
        state=$(squeue -j "$j" -h -t all -o "%t" 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')
        if [ -z "$state" ]; then
            state="(none)"
        else
            still_alive=$((still_alive + 1))
        fi
        prev="${LAST_STATE[$j]:-}"
        if [ "$state" != "$prev" ]; then
            log "  array $j  $prev → $state"
            LAST_STATE["$j"]="$state"
        fi
    done

    # new outputs
    partials=0
    for d in "${RESULTS_DIRS[@]}"; do
        [ -d "$d" ] || continue
        c=$(find "$d" -maxdepth 1 -name "dataset_*_r*_f*.pkl" 2>/dev/null | wc -l)
        partials=$((partials + c))
    done
    if [ "$partials" -ne "$PREV_PARTIALS" ]; then
        log "  partial pickles total: $PREV_PARTIALS → $partials"
        PREV_PARTIALS=$partials
    fi

    if [ "$still_alive" = "0" ] && [ "$TOTAL" = "0" ]; then
        log "all tracked arrays finished and queue is empty; exiting"
        break
    fi

    sleep "$POLL_SECS"
done

log "=== overnight monitor END ==="
