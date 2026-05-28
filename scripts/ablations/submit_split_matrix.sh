#!/usr/bin/env bash
# Submit the full split job matrix for the user-requested datasets.
#
# Per (dataset, noise) cell, submits one 100-task array (4 algorithms ×
# 5 repeats × 5 folds) then a merge+evaluate job that depends on the
# whole array via afterany.
#
# Total: 3 datasets × 4 noise × (100 array tasks + 1 merge) = 1200 + 12
# = 1212 SLURM jobs.

set -euo pipefail

cd "$(dirname "$0")/../.."

DATASETS=(
    "medical|results/full_medical_split|96G|2-00:00:00"
    "chestxray_densenet|results/full_chestxray_densenet_split|48G|1-00:00:00"
    "chestxray_resnet|results/full_chestxray_resnet_split|48G|1-00:00:00"
)
NOISES=(0.0 0.1 0.2 0.3)

SPLIT_SBATCH="scripts/ablations/full_train_split.sbatch"
MERGE_SBATCH="scripts/ablations/merge_and_eval.sbatch"

ALL_ARRAY_JIDS=()

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r DATASET RESULTS_DIR MEM TIME <<< "$entry"
    mkdir -p "${RESULTS_DIR}"
    for NOISE in "${NOISES[@]}"; do
        echo "=== submitting ${DATASET} noise=${NOISE} → ${RESULTS_DIR} ==="
        SPLIT_JID=$(
            sbatch --parsable \
                --array=0-99 \
                --time="${TIME}" \
                --cpus-per-task=4 \
                --mem="${MEM}" \
                --export=ALL,DATASET="${DATASET}",NOISE_RATE="${NOISE}",RESULTS_DIR="${RESULTS_DIR}",BASE_LEARNER=RF,SOLVER=highs \
                "${SPLIT_SBATCH}"
        )
        echo "  split array job: ${SPLIT_JID} (100 tasks)"

        MERGE_JID=$(
            sbatch --parsable \
                --dependency="afterany:${SPLIT_JID}" \
                --export=ALL,DATASET="${DATASET}",NOISE_RATE="${NOISE}",RESULTS_DIR="${RESULTS_DIR}" \
                "${MERGE_SBATCH}"
        )
        echo "  merge+eval job:  ${MERGE_JID} (depends on ${SPLIT_JID})"
        ALL_ARRAY_JIDS+=("${SPLIT_JID}" "${MERGE_JID}")
    done
done

echo
echo "=== submitted ${#ALL_ARRAY_JIDS[@]} top-level jobs (12 arrays + 12 merges = ${#ALL_ARRAY_JIDS[@]}) ==="
echo "${ALL_ARRAY_JIDS[@]}"
