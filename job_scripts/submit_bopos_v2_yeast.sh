#!/bin/bash
#SBATCH -o /home/s2320437/WORK/preorder4MLC/logs/slurm-%x-%j.log
#SBATCH -e /home/s2320437/WORK/preorder4MLC/logs/slurm-%x-%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hxtruong6+hakusan@gmail.com

# Yeast-specific BOPOs rerun-v2 driver. Yeast OOMs at the default 16 CPUs /
# 32 GB; this script raises memory to 96 GB and lowers CPUs to 8 to fit.
# Submit with: sbatch -p DEF -J "yeast_v2" job_scripts/submit_bopos_v2_yeast.sh

DATASET=${1:-yeast}

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate research_preorder_mlc

echo "Running on: $(hostname) | $(date)"
echo "DATASET=$DATASET (yeast-special override)"

cd /home/s2320437/WORK/preorder4MLC

RESULTS_DIR=results/20260514_v2

python main.py --dataset "$DATASET" --results_dir "$RESULTS_DIR"
python evaluation_test.py --dataset "$DATASET" --results_dir "$RESULTS_DIR"

echo "DONE: $DATASET $(date)"
