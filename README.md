# preorder4MLC: Pre-Order Based Multi-Label Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)

In **multi-label classification (MLC)** each instance can carry several
labels at once (a song that is both *happy* and *relaxed* — as in the
`emotions` dataset; an email tagged both *business* and *legal* — as in the
`enron` dataset, both used in this paper). This repository introduces a method that predicts
labels by first learning **how labels compare** rather than deciding each
label in isolation — and that can **abstain** on labels it is unsure about
instead of guessing.

> 📄 Code and full experimental results for:
> **\<TODO: paper title\>** — Hoàng Xuân Trường, Vu-Linh Nguyen.
> *Machine Learning* (Springer), 2026. \<TODO: DOI once assigned\>

## What it does

The method is built around **Bipartite Ordered Preference Orders (BOPOs)**.
Instead of predicting each label independently, it works in three steps:

1. **Learn pairwise preferences.** For every pair of labels, a calibrated
   classifier estimates the probability of their relative ordering.
2. **Search for the best order.** Per instance, an integer linear program
   (ILP) combines those pairwise probabilities into a single coherent
   **preference order** — a *pre-order* or a *partial-order* — that
   minimises an expected loss (Hamming or Subset 0/1).
3. **Derive a prediction.** The order is turned into one of three outputs:
   a plain binary vector, the preference order itself, or a
   **partial-abstention** vector that marks uncertain labels as "abstain".

An optional *height* constraint on the order yields **eight inference
algorithms** in total (pre-/partial-order × Hamming/Subset × height 2/∅).
We compare against four standard baselines: **BR, CC, CLR, ECC**.

```
              ┌──────────────────────┐
   training → │ K(K−1)/2 pairwise    │ → pairwise probabilities pᵢⱼ
              │ calibrated classif.  │
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ per-instance ILP     │ → preference order
              │ search (cvxopt+GLPK) │   (pre- or partial-order)
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ derive prediction    │ → BinaryVector
              │                      │   PreferenceOrder
              │                      │   PartialAbstention
              └──────────────────────┘
```

## Quick start

```bash
# Install (editable, so scripts/ can import the package)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run one dataset end-to-end
python scripts/train.py    --dataset emotions --results_dir results/run-dev
python scripts/evaluate.py --dataset emotions --results_dir results/run-dev
```

To reproduce **every** number and figure in the paper, see
[REPRODUCE.md](REPRODUCE.md) — it covers the full pipeline, expected wall
times, and the exact environment.

## Datasets

Ten multi-label datasets are used:
`chd_49`, `emotions`, `scene`, `yeast`, `water_quality`, `humanpseaac`,
`gpositivepseaac`, `plantpseaac`, `viruspseaac`, and `enron`.

The first nine ARFFs are bundled under `data/`, so the pipeline runs
end-to-end right after `pip install`. `enron.arff` (K=53) is downloaded
separately — see [REPRODUCE.md §2](REPRODUCE.md). CLI keys match
`preorder4mlc.config::ConfigManager.DATASET_CONFIGS`.

## Partial abstention

A partial-abstention prediction is a vector in `{0, 1, −1}` where `−1`
means **"abstain"** on that label. This lets the model stay silent where
it is uncertain instead of forcing a 0/1 call. Two pairs of metrics
capture the trade-off — *recovery* (did the abstentions cover the truth?)
and *abstention rate* (how often did it abstain?):

```
ŷ = [1, 0, 1, −1, 0, −1]      y = [0, 0, 1, 1, 0, 0]

AREC = (0 + 1 + 1 + 1 + 1 + 1) / 6 = 4/6      # −1 counts as covering {0,1}
AABS = 2 / 6                                   # fraction abstained
```

Full definitions (`AREC`, `AABS`, `REC`, `ABS`) live in
`preorder4mlc.evaluation_metric`.

## Repository layout

```
preorder4mlc/            # Library package
├── config.py            # Run configuration + dataset registry
├── datasets4experiments.py   # ARFF loading, k-fold splits, label noise
├── base_classifiers.py  # Pairwise / calibrated classifier factory
├── estimator.py         # Uniform interface over RF / ETC / XGBoost / LightGBM
├── inference_models.py  # PredictBOPOs (BOPOs + BR / CC / CLR baselines)
├── searching_algorithms.py   # ILP search for pre- and partial-orders
├── training_orchestrator.py  # Training loop over learners × folds × algorithms
├── evaluation_metric.py # Example-, label-, ranking-, abstention-metrics
└── utils/               # Summaries, statistical tests, figures

scripts/                 # CLI entry points
├── train.py / evaluate.py          # BOPOs + CLR / BR / CC
├── train_ecc.py / evaluate_ecc.py  # ECC baseline
└── smoke_predict_bopos.py          # Behavior-preservation smoke test

data/                    # 9 bundled ARFFs (enron downloaded separately)
results/                 # Per-fold CSVs + aggregated tables
run.sh                   # End-to-end reproduction driver
REPRODUCE.md             # Step-by-step reproduction recipe
```

## Citation

If you use this code, please cite the paper (see
[CITATION.cff](CITATION.cff)):

```bibtex
@article{TODO,
  title     = {<TODO: paper title>},
  author    = {Hoàng, Xuân Trường and Nguyen, Vu-Linh},
  journal   = {Machine Learning},
  publisher = {Springer},
  year      = {2026},
}
```

## License

Released under the [MIT License](LICENSE).
