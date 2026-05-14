#!/bin/bash
#SBATCH -o /home/s2320437/WORK/preorder4MLC/logs/slurm-%x-%j.log
#SBATCH -e /home/s2320437/WORK/preorder4MLC/logs/slurm-%x-%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hxtruong6+hakusan@gmail.com

# BOPOs (+ BR/CC/CLR) rerun-v2 driver, parameterized by dataset.
# Submit with: sbatch -p DEF -J "<dataset>_v2" job_scripts/submit_bopos_v2.sh <dataset>

DATASET=${1:?Usage: <DATASET>}

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate research_preorder_mlc

echo "Running on: $(hostname) | $(date)"
echo "DATASET=$DATASET"

cd /home/s2320437/WORK/preorder4MLC

RESULTS_DIR=results/20260514_v2

python main.py --dataset "$DATASET" --results_dir "$RESULTS_DIR"
python evaluation_test.py --dataset "$DATASET" --results_dir "$RESULTS_DIR"

echo "DONE: $DATASET $(date)"
