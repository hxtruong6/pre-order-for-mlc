#!/bin/bash
#SBATCH -o /home/s2320437/WORK/preorder4MLC/logs/slurm-%x-%j.log
#SBATCH -e /home/s2320437/WORK/preorder4MLC/logs/slurm-%x-%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL

# Final eval pass — runs evaluation_test.py once after all 4 noise jobs finish.
# Submit with: sbatch -p DEF -J "<dataset>_eval_v2" --dependency=afterok:j1:j2:j3:j4 submit_eval_after.sh <dataset>

DATASET=${1:?Usage: <DATASET>}

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate research_preorder_mlc

echo "Running on: $(hostname) | $(date)"
echo "DATASET=$DATASET"

cd /home/s2320437/WORK/preorder4MLC

RESULTS_DIR=results/20260514_v2

python evaluation_test.py --dataset "$DATASET" --results_dir "$RESULTS_DIR"

echo "DONE: $DATASET eval $(date)"
