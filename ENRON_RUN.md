# Running ENRON (K=53) on a New SLURM HPC

End-to-end guide to reproduce the preorder4MLC pipeline on the enron dataset
on a fresh HPC. Assumes the cluster runs SLURM and provides standard CPU
compute nodes (no GPU required).

## 1. Repo + environment

```bash
# Clone (or rsync from the original cluster)
git clone <repo-url> preorder4MLC
cd preorder4MLC
git checkout feat/perf-large-k-optimizations   # branch with all the K=45+ fixes

# Python venv (3.11 recommended — matches the original)
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                                # editable install of the package

# Sanity smoke check (must print the exact hash below)
PYTHONPATH=. python scripts/smoke_predict_bopos.py --out_dir /tmp/smoke
# Expected: sha256 4603455317bd74a072b7f0ffab22a8a5e3c9d8c6a76b213097e155af87709244
```

If the hash differs, do NOT proceed — the inference pipeline diverges from the
reference and downstream evaluation will be invalid.

## 2. Data

Place the enron ARFF file at `./data/enron.arff` (53 labels, target columns at
the *start* of each row — this is the default in
`Datasets4Experiments.load_datasets`; do not add `enron.arff` to
`TARGET_IN_END_FILE_DATASETS`).

```bash
ls -la data/enron.arff   # must exist before submitting
```

Verify the dataset key is registered:

```bash
grep -A1 '"enron"' preorder4mlc/config.py
# → "enron": DatasetConfig("enron", "enron.arff", 53),
```

## 3. Algorithms to run

For each dataset there are **7 algorithms**:

| Group | Algorithm | Entry script | Split strategy |
|---|---|---|---|
| Main | BOPOS | `scripts/train.py` | per (repeat, fold) — **slow** |
| Main | CLR   | `scripts/train.py` | per (repeat, fold) |
| Main | BR    | `scripts/train.py` | per (repeat, fold) |
| Main | CC    | `scripts/train.py` | per (repeat, fold) |
| Extra | MLkNN | `scripts/train_extra_baselines.py` | one job, sweeps 4 noise × 25 (R,F) |
| Extra | ECC   | `scripts/train_extra_baselines.py` | one job, sweeps 4 noise × 25 (R,F) |
| Extra | LP    | `scripts/train_extra_baselines.py` | one job, sweeps 4 noise × 25 (R,F) |

The matrix per dataset is 4 noise levels × 25 (R,F) × 4 main algos = **400
SLURM array tasks** + 3 extra-baseline jobs + 4 merge jobs.

## 4. SLURM scripts (already in the repo)

```
scripts/ablations/full_train_split.sbatch      # array body for main 4 algos
scripts/ablations/extra_baseline.sbatch        # body for mlknn/ecc/lp
scripts/ablations/merge_and_eval.sbatch        # merge partials + evaluate
scripts/ablations/merge_split_results.py       # consolidator
scripts/ablations/master_controller.py         # poll-and-submit loop
scripts/ablations/submit_extras.sh             # submit extras (one-shot)
```

`full_train_split.sbatch` reads these env vars:

| Var | Purpose | Default |
|---|---|---|
| `DATASET`      | dataset key (required) | — |
| `NOISE_RATE`   | one of 0.0, 0.1, 0.2, 0.3 (required) | — |
| `RESULTS_DIR`  | output dir (required) | — |
| `ALGORITHM`    | bopos / clr / br / cc (required) | — |
| `BASE_LEARNER` | RF / ETC / XGBoost / LightGBM | `RF` |
| `SOLVER`       | **highs** or glpk (see §7) | `highs` |

`SLURM_ARRAY_TASK_ID` is mapped to (repeat, fold) via `repeat = id // 5`,
`fold = id % 5`. Submit with `--array=0-24` (25 tasks per algorithm).

## 5. One-shot submission for enron

Three options, pick what fits your QOS limits:

### Option A — master controller (recommended)

Add an "enron" entry to `master_controller.py`:

```python
# scripts/ablations/master_controller.py, in DATASETS:
"enron": {
    "results_dir": "results/full_enron_split",
    "mem": "160G",     # K=53 BOPOS needs > 96G; see §7
    "time": "2-00:00:00",
},
```

Adjust `PREORDER_CAP` to match your cluster's per-user MaxJobs limit (we used
36 for a 40-job cap). Then run:

```bash
mkdir -p results/full_enron_split slurm_logs
POLL_SECS=900 nohup python scripts/ablations/master_controller.py \
    > slurm_logs/master_controller.log 2>&1 &
echo $! > /tmp/master_controller.pid
```

The controller scans disk every `POLL_SECS` seconds, submits enough array
tasks to fill the queue, and auto-triggers `merge_and_eval.sbatch` when a
(dataset, noise) cell reaches 100/100 partials.

### Option B — single dataset, no controller

```bash
RESULTS_DIR=results/full_enron_split
mkdir -p "$RESULTS_DIR"

for NOISE in 0.0 0.1 0.2 0.3; do
  for ALGO in bopos clr br cc; do
    sbatch --array=0-24 --time=2-00:00:00 --cpus-per-task=4 --mem=160G \
      --export=ALL,DATASET=enron,NOISE_RATE=$NOISE,RESULTS_DIR=$RESULTS_DIR,ALGORITHM=$ALGO,BASE_LEARNER=RF,SOLVER=highs \
      scripts/ablations/full_train_split.sbatch
  done
done

# After all 16 arrays finish, merge + evaluate each cell:
for NOISE in 0.0 0.1 0.2 0.3; do
  sbatch --dependency=afterany:... --export=ALL,DATASET=enron,NOISE_RATE=$NOISE,RESULTS_DIR=$RESULTS_DIR \
    scripts/ablations/merge_and_eval.sbatch
done
```

### Option C — extras

```bash
for ALGO in mlknn ecc lp; do
  sbatch --time=2-00:00:00 --cpus-per-task=4 --mem=160G \
    --export=ALL,DATASET=enron,RESULTS_DIR=results/full_enron_split,ALGO=$ALGO,BASE_LEARNER=rf \
    scripts/ablations/extra_baseline.sbatch
done
```

`train_extra_baselines.py` iterates all 4 noise levels and all 25 (R,F) inside
one job — one submission per (dataset, algorithm), 3 total for enron.

## 6. Resource estimate for K=53

Measured on medical (K=45) and extrapolated by O(K²) scaling (pairwise
classifiers = K(K-1)/2). enron K=53 → 1378 pairs vs medical's 990 (1.39×).

| Algo | medical (per task) | enron estimate (per task) |
|---|---|---|
| BOPOS | 60 min | **85–100 min** |
| CLR   | 33 s   | ~45 s |
| BR    | 9 s    | ~13 s |
| CC    | 9 s    | ~13 s |

With 25-task parallel (one per R,F), one (dataset, noise) cell wall-clock is
bounded by the slowest BOPOS task ≈ **~100 min**. Full sweep (4 noise) ≈
**6–7 hours wall-clock** if 25 BOPOS tasks can run concurrently.

Memory: bump to **160G per task** for BOPOS. medical n=0.2 OOMed at 96G in
our run; K=53 needs at least the same.

Time limit: `--time=2-00:00:00` (2 days) is safe headroom.

CPU: `--cpus-per-task=4` is enough — joblib parallelism is bounded by
`LOKY_MAX_CPU_COUNT=4` inside the sbatch wrapper.

## 7. Critical gotchas

### a. Use SOLVER=highs, not glpk

The ILP search in `searching_algorithms.py` supports two backends.
`cvxopt.glpk.ilp` (the legacy default) can hang for hours on K≥45 problems —
this was the reason enron jobs ran 38h+ without producing output on the
previous cluster. **Always pass `SOLVER=highs`** (`scipy.optimize.milp`). This
is the new default in `full_train_split.sbatch` but pass it explicitly to be
sure.

### b. Bug fixes already applied (do not revert)

The branch `feat/perf-large-k-optimizations` includes three fixes for
degenerate CV folds that appear on K≥45 datasets (medical, enron):

- `inference_models.py:193` — CLR broadcast: `[:, 0]` slice on single-class
  calibrated classifier output.
- `base_classifiers.py` — skip empty pairs (both labels always equal in
  train) before parallel RF fit.
- `inference_models.py:_BinarySafeClassifier` — wraps the CC base classifier
  so `ClassifierChain.predict_proba` returns shape (n, 2) even when a chain
  step is trained on single-class data.

Smoke hash above is preserved by all three fixes.

### c. Per-user QOS / MaxJobs cap

Check your cluster's limit before running the controller:

```bash
sacctmgr -P show qos --noheader | head -5      # find your QOS row
scontrol show partition DEF | grep -i max      # partition-level
squeue -u $USER -h -t PD -o "%R" | sort -u     # reasons jobs stay pending
```

Set `PREORDER_CAP` in `master_controller.py` to (cluster_max_jobs - 5) to
leave buffer for extras + merge jobs.

### d. State recovery

Controller state is at `/tmp/preorder_master_state.json`. To restart cleanly:

```bash
kill $(cat /tmp/master_controller.pid)
rm /tmp/preorder_master_state.json             # optional: rescan from scratch
POLL_SECS=900 nohup python scripts/ablations/master_controller.py \
    > slurm_logs/master_controller.log 2>&1 &
```

On restart the controller scans `results/full_enron_split/` for existing
`dataset_enron_noisy_*_r*_f*.pkl` files and only submits the missing ones —
safe to restart at any point.

## 8. Outputs

After all tasks succeed each `(noise)` cell produces:

```
results/full_enron_split/dataset_enron_noisy_<r>_r<R>_f<F>.pkl          # raw partials
results/full_enron_split/dataset_enron_noisy_<r>_<algo>_r<R>_f<F>.pkl   # baselines
results/full_enron_split/dataset_enron_noisy_<r>.pkl                    # merged BOPOS
results/full_enron_split/dataset_enron_noisy_<r>_clr.pkl                # merged CLR
… etc
results/full_enron_split/evaluation_*.csv                               # eval tables
```

Final summary across noise levels:

```bash
python -m preorder4mlc.utils.summarize_metrics \
    --results_dir results/full_enron_split \
    --output_dir  results/full_enron_split_summary
```

## 9. Quick checklist

- [ ] Branch `feat/perf-large-k-optimizations` checked out
- [ ] `data/enron.arff` present
- [ ] Smoke hash matches
- [ ] `master_controller.py` has an `enron` entry with `mem=160G`
- [ ] `PREORDER_CAP` matches your cluster's per-user limit
- [ ] `SOLVER=highs` explicitly set
- [ ] `slurm_logs/` directory exists
- [ ] Controller launched and `master_controller_report.txt` shows enron rows

## Contact / Reference

- Original cluster run: 3 datasets (medical, chestxray_densenet,
  chestxray_resnet) at 4 noise levels using this same pipeline; ~1200 main
  tasks + 9 extras completed in ~24h on a 40-jobs-per-user QOS.
- Repo: see `CLAUDE.md` for full architecture notes.
