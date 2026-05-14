#!/bin/bash
#SBATCH -o /home/s2320437/WORK/preorder4MLC/logs/slurm-%x-%j.log
#SBATCH -e /home/s2320437/WORK/preorder4MLC/logs/slurm-%x-%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hxtruong6+hakusan@gmail.com

# Full Yeast run: train BOPOs + baselines, then evaluate.
# Submit with: sbatch -p DEF -J yeast submit_yeast.sh

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate research_preorder_mlc

echo "Running on: $(hostname) | $(date)"

cd /home/s2320437/WORK/preorder4MLC

RESULTS_DIR=results/20250624

python main.py --dataset yeast --results_dir "$RESULTS_DIR"
python evaluation_test.py --dataset yeast --results_dir "$RESULTS_DIR"

echo "DONE: yeast $(date)"
