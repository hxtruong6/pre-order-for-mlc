#!/bin/bash
#SBATCH --job-name=glpk_pre_full_k53
#SBATCH --partition=DEF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/home/s2320437/glpk_pre_k53_run/slurm_%j.out
#SBATCH --error=/home/s2320437/glpk_pre_k53_run/slurm_%j.out

# Measure GLPK pre-order FULL-transitivity at K=53 for both target metrics.
# Each solve is single-threaded (GLPK branch-and-cut), so the two configs run
# as two parallel processes. Results land in $PREORDER_BENCH_DIR as
# glpk_k53_preorder_p_full_{Hamming,Subset}.csv, which bench_step8_tikz.py
# globs. --time=48:00:00 is a hard cap: if a solve does not finish, SLURM
# kills the job and the corresponding CSV is simply absent (-> n/a in table).

set -u
export PREORDER_BENCH_DIR=/home/s2320437/glpk_pre_k53_run
export PYTHONPATH=/home/s2320437/WORK/preorder4MLC
# keep each GLPK solve to a single core (no BLAS oversubscription)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1

PY=/home/s2320437/miniconda3/envs/research_preorder_mlc/bin/python
cd /home/s2320437/WORK/preorder4MLC

echo "[$(date)] host=$(hostname) starting GLPK pre-order full K=53 (2 configs)"

$PY scripts/bench_step10_one.py Hamming full &
PID_H=$!
$PY scripts/bench_step10_one.py Subset  full &
PID_S=$!

wait $PID_H; echo "[$(date)] Hamming full exit=$?"
wait $PID_S; echo "[$(date)] Subset  full exit=$?"
echo "[$(date)] DONE"
