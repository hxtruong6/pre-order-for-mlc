#!/bin/bash
#SBATCH -o /home/s2320437/WORK/preorder4MLC/logs/slurm-%x-%j.log
#SBATCH -e /home/s2320437/WORK/preorder4MLC/logs/slurm-%x-%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

DATASET=${1:?Usage: <DATASET> <ALGO>}
ALGO=${2:?Usage: <DATASET> <ALGO>}

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate research_preorder_mlc

echo "Running on: $(hostname) | $(date)"
echo "DATASET=$DATASET ALGO=$ALGO"

cd /home/s2320437/WORK/preorder4MLC

RESULTS_DIR=results/20250624

python extra_baselines.py --dataset "$DATASET" --algorithm "$ALGO" --results_dir "$RESULTS_DIR"
python evaluate_extra_baselines.py --dataset "$DATASET" --algorithm "$ALGO" --results_dir "$RESULTS_DIR"

echo "DONE: $DATASET $ALGO $(date)"
