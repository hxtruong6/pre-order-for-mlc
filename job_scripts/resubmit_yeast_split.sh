#!/bin/bash
# Resubmit yeast BOPOs as 4 parallel per-noise jobs + 1 dependent eval job.
# Usage: bash job_scripts/resubmit_yeast_split.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITION="${PARTITION:-DEF}"
DATASET="yeast"

JIDS=()
for noise in 0.0 0.1 0.2 0.3; do
    jid=$(sbatch -p "$PARTITION" -J "${DATASET}_n${noise}_v2" \
        "${SCRIPT_DIR}/submit_bopos_v2_split.sh" "$DATASET" "$noise" \
        | awk '{print $NF}')
    echo "[bopos]    dataset=$DATASET noise=$noise  JOBID=$jid"
    JIDS+=("$jid")
done

DEP=$(IFS=:; echo "${JIDS[*]}")
eval_jid=$(sbatch -p "$PARTITION" -J "${DATASET}_eval_v2" \
    --dependency=afterok:$DEP \
    "${SCRIPT_DIR}/submit_eval_after.sh" "$DATASET" \
    | awk '{print $NF}')
echo "[eval]     dataset=$DATASET deps=$DEP  JOBID=$eval_jid"
