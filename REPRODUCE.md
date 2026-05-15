# Reproducing the paper results

This document records the exact steps required to reproduce every number
and figure reported in the paper from a clean clone of this repository.

## 1. Environment

Tested on Python 3.11 with the pinned versions in `requirements.txt`.
GLPK headers must be present on the system so that `cvxopt.glpk` can
solve the per-instance ILP.

```bash
# Ubuntu / Debian
sudo apt-get install libglpk-dev

# Python deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Data

The 9 multi-label ARFF datasets used by the paper are tracked under
`data/` for one-click reproduction:

| Key (CLI) | File | Labels | Source |
|---|---|---|---|
| `chd_49` | `CHD_49.arff` | 6 | Coronary heart disease |
| `emotions` | `emotions.arff` | 6 | Mulan repository |
| `scene` | `scene.arff` | 6 | Mulan repository |
| `yeast` | `Yeast.arff` | 14 | Mulan repository |
| `water_quality` | `Water-quality.arff` | 14 | UCI |
| `viruspseaac` | `VirusPseAAC.arff` | 6 | Pse-AAC encoding |
| `humanpseaac` | `HumanPseAAC.arff` | 14 | Pse-AAC encoding |
| `gpositivepseaac` | `GpositivePseAAC.arff` | 4 | Pse-AAC encoding |
| `plantpseaac` | `PlantPseAAC.arff` | 12 | Pse-AAC encoding |

## 3. Determinism

All splits, folds, and label-noise draws are seeded by
`preorder4mlc.constants.RANDOM_STATE = 6`. The per-fold parallel
training step uses `joblib.Parallel(n_jobs=-1)`; sklearn ensemble
estimators are seeded through `RANDOM_STATE` as well so identical
splits produce identical predictions.

## 4. Reproducing all results

A single command reproduces the full pipeline for all 9 datasets:

```bash
RESULTS_DIR=results/20260514_v2 bash run.sh
```

For each dataset this runs, in order:

1. `python scripts/train.py --dataset <key> --results_dir <dir>` —
   train pairwise classifiers, BOPOs (pre- and partial-order), and
   CLR / BR / CC baselines.
2. `python scripts/evaluate.py --dataset <key> --results_dir <dir>` —
   write per-fold evaluation CSVs for the BOPOs / CLR / BR / CC pickles.
3. `python scripts/train_extra_baselines.py --dataset <key>
   --algorithm <mlknn|ecc|lp> --results_dir <dir>` — train the three
   extra baselines.
4. `python scripts/evaluate_extra_baselines.py --dataset <key>
   --algorithm <mlknn|ecc|lp> --results_dir <dir>` — write per-fold
   evaluation CSVs for the extra baselines.

Per-dataset stdout/stderr lands in `logs/20260514_v2/<dataset>.log`.

## 5. Summarising into the paper tables

```bash
python -m preorder4mlc.utils.summarize_metrics \
    --results_dir results/20260514_v2 \
    --output_dir  results/final_20260514_v2_summary
```

Writes one `<Dataset>_<PredictionType>_summary.csv` per dataset to
`results/final_20260514_v2_summary/`. The `PredictionType` suffix is
one of `BinaryVector`, `PartialAbstention`, or `ScoreVector`.

## 6. Statistical tests and figures

```bash
python -m preorder4mlc.utils.statistical_tests \
    --results_dir results/final_20260514_v2_summary \
    --output_dir  results/final_20260514_v2_summary/stats

python -m preorder4mlc.utils.plot_figures \
    --results_dir     results/final_20260514_v2_summary \
    --raw_results_dir results/20260514_v2
```

The figure script writes critical-difference diagrams, average-rank vs
noise curves, abstention plots, Hamming/Subset trade-offs, and rank
heatmaps to `results/final_20260514_v2_summary/figures/`.

## 7. Compute budget

Single-machine wall time on the v2 run (2 repeats x 5 folds, joblib
across all CPUs), per dataset:

| Dataset | Wall time |
|---|---|
| chd_49 | ~4 min |
| viruspseaac | ~4 min |
| gpositivepseaac | ~5 min |
| emotions | ~6 min |
| plantpseaac | ~30 min |
| water_quality | ~60 min |
| scene | ~40 min |
| yeast | ~60 min |
| humanpseaac | ~2.5 h |

Total wall time: roughly 6 hours end-to-end, with humanpseaac being the
bottleneck. The three extra baselines per dataset add at most ~30 min
each (typically far less). For HPC re-runs, dataset jobs can be
submitted in parallel.

## 8. Output bundle expected by the paper

After step 6 the repository contains:

* `results/20260514_v2/*.pkl` — per-fold training records (not tracked
  by git; recreated by step 4).
* `results/20260514_v2/evaluation_*.csv` and `*.xlsx` — per-fold
  metric CSVs (tracked by git).
* `results/final_20260514_v2_summary/*_summary.{csv,xlsx}` — per-dataset
  aggregate tables (tracked).
* `results/final_20260514_v2_summary/stats/` — Friedman /
  Nemenyi outputs and CD-diagram PDFs.
* `results/final_20260514_v2_summary/figures/` — paper figures.
