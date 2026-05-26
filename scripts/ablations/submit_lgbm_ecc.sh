#!/bin/bash
# Submit ECC extra-baseline arrays for all LightGBM dirs still missing them.
# Throttles to a 28-task preorder-* cap (def QOS).
# Each cell = (dataset, noise) → 1 array of 5 tasks (R=0, F=0-4) since LGBM
# is special-cased to one repeat in the orchestrator.
set -uo pipefail
cd /home/s2320437/WORK/preorder4MLC

# (dataset_arg, results_dir) — dataset arg is what train_extra_baselines.py
# expects (lowercase or with hyphen for water-quality).
PAIRS=(
  "chd_49 results/full_chd_49_lgbm_split"
  "emotions results/full_emotions_lgbm_split"
  "gpositivepseaac results/full_gpositivepseaac_lgbm_split"
  "humanpseaac results/full_humanpseaac_lgbm_split"
  "plantpseaac results/full_plantpseaac_lgbm_split"
  "scene results/full_scene_lgbm_split"
  "viruspseaac results/full_viruspseaac_lgbm_split"
  "water_quality results/full_water_quality_lgbm_split"
  "yeast results/full_yeast_lgbm_split"
)
NOISES=(0.0 0.1 0.2 0.3)
CAP=28

count_preorder() {
  squeue -u "$USER" -r -h -o "%j" 2>/dev/null | grep -c '^preorder-' || echo 0
}

submit_cell() {
  local ds="$1" rdir="$2" noise="$3"
  sbatch --parsable \
    --array=0-4 \
    --time=02:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --export=ALL,DATASET="${ds}",ALGO=ecc,NOISE_RATE="${noise}",RESULTS_DIR="${rdir}",BASE_LEARNER=lgbm \
    scripts/ablations/extra_split.sbatch 2>&1
}

mkdir -p slurm_logs
LOG=slurm_logs/submit_lgbm_ecc.log

for pair in "${PAIRS[@]}"; do
  read -r DS RDIR <<< "$pair"
  for NOISE in "${NOISES[@]}"; do
    # Skip if all 5 partials already exist for this cell
    n=$(ls "${RDIR}/dataset_${DS}_noisy_${NOISE}_ecc_lgbm_r"*"_f"*.pkl 2>/dev/null | wc -l)
    if [ "$n" -ge 5 ]; then
      echo "[$(date +%H:%M:%S)] SKIP ${DS}/n=${NOISE} (already 5 partials)" | tee -a "$LOG"
      continue
    fi

    # Throttle: wait until < CAP
    while true; do
      cur=$(count_preorder)
      if [ "$cur" -lt "$((CAP - 4))" ]; then
        break
      fi
      sleep 30
    done

    jid=$(submit_cell "${DS}" "${RDIR}" "${NOISE}")
    echo "[$(date +%H:%M:%S)] submitted ${DS}/n=${NOISE} ecc -> ${jid}" | tee -a "$LOG"
    sleep 2
  done
done

echo "[$(date +%H:%M:%S)] === all ECC arrays submitted ===" | tee -a "$LOG"
