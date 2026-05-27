# Reproducing the paper results

This document records the exact steps required to reproduce every number
and figure reported in the journal-revision paper from a clean clone of
this repository.

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

Nine multi-label datasets are used in the paper. Eight ARFFs are tracked
under `data/` for one-click reproduction; `enron.arff` (K=53) must be
downloaded separately from COMETA / MULAN and placed at
`data/enron.arff`.

| Key (CLI) | File | Labels | Source |
|---|---|---|---|
| `chd_49` | `CHD_49.arff` | 6 | Coronary heart disease |
| `emotions` | `emotions.arff` | 6 | Mulan repository |
| `scene` | `scene.arff` | 6 | Mulan repository |
| `yeast` | `Yeast.arff` | 14 | Mulan repository |
| `water_quality` | `Water-quality.arff` | 14 | UCI |
| `humanpseaac` | `HumanPseAAC.arff` | 14 | Pse-AAC encoding |
| `gpositivepseaac` | `GpositivePseAAC.arff` | 4 | Pse-AAC encoding |
| `plantpseaac` | `PlantPseAAC.arff` | 12 | Pse-AAC encoding |
| `enron` | `enron.arff` | 53 | COMETA / MULAN (not bundled) |

## 3. Determinism

All splits, folds, and label-noise draws are seeded by
`preorder4mlc.constants.RANDOM_STATE = 6`. The per-fold parallel
training step uses `joblib.Parallel(n_jobs=-1)`; sklearn ensemble
estimators are seeded through `RANDOM_STATE` as well so identical
splits produce identical predictions.

## 4. Reproducing all results

A single command reproduces the full pipeline for all nine datasets
with the default base learner (Random Forest):

```bash
RESULTS_DIR=results/run-$(date +%Y%m%d) bash run.sh
```

For each dataset this runs, in order:

1. `python scripts/train.py --dataset <key> --results_dir <dir>` —
   train pairwise classifiers, BOPOs (pre- and partial-order), and the
   CLR / BR / CC baselines.
2. `python scripts/evaluate.py --dataset <key> --results_dir <dir>` —
   write per-fold evaluation CSVs for the BOPOs / CLR / BR / CC pickles.
3. `python scripts/train_ecc.py --dataset <key> --algorithm ecc
   --results_dir <dir>` — train the ECC (Ensemble of Classifier Chains)
   baseline.
4. `python scripts/evaluate_ecc.py --dataset <key> --algorithm ecc
   --results_dir <dir>` — write per-fold evaluation CSVs for ECC.

Per-dataset stdout/stderr lands in `logs/run-<date>/<dataset>.log`.

To reproduce the LightGBM variant of the paper, pass
`--base_learner lgbm` to both `train.py` and `train_ecc.py`.

## 5. Summarising into the paper tables

```bash
python -m preorder4mlc.utils.summarize_metrics \
    --results_dir results/run-<date> \
    --output_dir  results/run-<date>_summary
```

Writes one `<Dataset>_<PredictionType>_summary.csv` per dataset to the
output directory. `PredictionType` is one of `BinaryVector`,
`PartialAbstention`, or `ScoreVector`.

## 6. Statistical tests and figures

```bash
python -m preorder4mlc.utils.statistical_tests \
    --results_dir results/run-<date>_summary \
    --output_dir  results/run-<date>_summary/stats

python -m preorder4mlc.utils.plot_figures \
    --results_dir     results/run-<date>_summary \
    --raw_results_dir results/run-<date>
```

The figure script writes critical-difference diagrams, average-rank vs
noise curves, abstention plots, Hamming/Subset trade-offs, and rank
heatmaps to `results/run-<date>_summary/figures/`.

## 7. Compute budget

Single-machine wall time (5 repeats x 5 folds, joblib across all CPUs),
per dataset, with Random Forest as the base learner:

| Dataset | Wall time |
|---|---|
| chd_49 | ~4 min |
| gpositivepseaac | ~5 min |
| emotions | ~6 min |
| plantpseaac | ~30 min |
| scene | ~40 min |
| water_quality | ~60 min |
| yeast | ~60 min |
| humanpseaac | ~2.5 h |
| enron | ~12 h (K=53; recommend an HPC node with 32 GB RAM) |

ECC adds at most ~30 min per dataset (typically far less). For HPC
re-runs, dataset jobs can be submitted in parallel.

## 8. Output bundle

After step 6 the run directory contains:

* `results/run-<date>/*.pkl` — per-fold training records (not tracked
  by git; recreated by step 4).
* `results/run-<date>/evaluation_*.csv` and `*.xlsx` — per-fold
  metric CSVs.
* `results/run-<date>_summary/*_summary.{csv,xlsx}` — per-dataset
  aggregate tables consumed by `statistical_tests` and `plot_figures`.
* `results/run-<date>_summary/stats/` — Friedman / Nemenyi outputs
  and CD-diagram PDFs.
* `results/run-<date>_summary/figures/` — paper figures.

The canonical Random-Forest aggregate tables that produced the paper
numbers ship under `results/final_20260514_v2_summary/`, and the
LightGBM aggregate tables ship under
`results/full_<dataset>_lgbm_summary/`. New runs can be diffed against
these to confirm byte-for-byte equivalence.
