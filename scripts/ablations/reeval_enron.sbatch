#!/bin/bash
#SBATCH --job-name=reeval-enron
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Re-evaluate enron with the updated evaluator (adds jaccard_pa, macro_f1_pa,
# micro_f1_pa and other newer PA metrics that were missing from the original
# run). All merged PKLs already exist in full_enron_split/ — no training needed.

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results/full_enron_split}"
SUMMARY_DIR="${SUMMARY_DIR:-results/full_enron_split_summary}"

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

echo "=== host=$(hostname) job=${SLURM_JOB_ID} ==="
echo "=== results=${RESULTS_DIR} summary=${SUMMARY_DIR} ==="

echo
echo "--- EVALUATE (all noise levels, all algorithms) ---"
time python scripts/evaluate.py \
    --dataset enron \
    --results_dir "${RESULTS_DIR}"

echo
echo "--- SUMMARIZE ---"
mkdir -p "${SUMMARY_DIR}"
time python -m preorder4mlc.utils.summarize_metrics \
    --results_dir "${RESULTS_DIR}" \
    --output_dir  "${SUMMARY_DIR}"

echo
echo "=== done ==="
echo "PA metrics now in summary:"
python -c "
import csv
with open('${SUMMARY_DIR}/enron_PartialAbstention_summary.csv') as f:
    cols = next(csv.reader(f))
metrics = sorted({c.rsplit('__',1)[0] for c in cols[1:] if '__' in c})
print('  ' + '\n  '.join(metrics))
"
