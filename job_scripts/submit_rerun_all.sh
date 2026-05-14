#!/bin/bash
# Orchestrator (plain bash, NOT a slurm script): submits the 9 BOPOs jobs +
# 27 baseline jobs for rerun-v2.
#
# For yeast we use the special yeast script (8 CPUs / 96 GB) to avoid OOM.
# For all other datasets we use submit_bopos_v2.sh (16 CPUs / 32 GB).
#
# Usage: bash job_scripts/submit_rerun_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITION="${PARTITION:-DEF}"

DATASETS=(
    chd_49
    emotions
    scene
    viruspseaac
    yeast
    water_quality
    humanpseaac
    gpositivepseaac
    plantpseaac
)

BASELINE_ALGOS=(mlknn ecc lp)

echo "[submit-rerun-all] Submitting BOPOs jobs (9) and baseline jobs (27)..."

for d in "${DATASETS[@]}"; do
    if [[ "$d" == "yeast" ]]; then
        bopos_script="${SCRIPT_DIR}/submit_bopos_v2_yeast.sh"
    else
        bopos_script="${SCRIPT_DIR}/submit_bopos_v2.sh"
    fi

    jid=$(sbatch -p "$PARTITION" -J "${d}_v2" "$bopos_script" "$d" | awk '{print $NF}')
    echo "[bopos]    dataset=$d  JOBID=$jid  script=$(basename "$bopos_script")"

    for algo in "${BASELINE_ALGOS[@]}"; do
        jid=$(sbatch -p "$PARTITION" -J "${d}_${algo}_v2" \
            "${SCRIPT_DIR}/submit_extra_baseline_v2.sh" "$d" "$algo" \
            | awk '{print $NF}')
        echo "[baseline] dataset=$d algo=$algo  JOBID=$jid"
    done
done

echo "[submit-rerun-all] Done. Use 'squeue -u \$USER' to monitor."
