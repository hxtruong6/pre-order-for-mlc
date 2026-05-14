#!/bin/bash
#SBATCH -o /home/s2320437/WORK/preorder4MLC/logs/slurm-%x-%j.log
#SBATCH -e /home/s2320437/WORK/preorder4MLC/logs/slurm-%x-%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --mail-type=FAIL

# BOPOs single-noise driver (no eval — eval runs in a final job after all 4 noise jobs).
# Submit with: sbatch -p DEF -J "<dataset>_n<noise>_v2" submit_bopos_v2_split.sh <dataset> <noise_rate>

DATASET=${1:?Usage: <DATASET> <NOISE_RATE>}
NOISE=${2:?Usage: <DATASET> <NOISE_RATE>}

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate research_preorder_mlc

echo "Running on: $(hostname) | $(date)"
echo "DATASET=$DATASET NOISE=$NOISE"

cd /home/s2320437/WORK/preorder4MLC

RESULTS_DIR=results/20260514_v2

python main.py --dataset "$DATASET" --results_dir "$RESULTS_DIR" --noise_rate "$NOISE"

echo "DONE: $DATASET noise=$NOISE $(date)"
