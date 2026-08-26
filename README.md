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
> **Robust multi-label classification via preference learning** —
> Vu-Linh Nguyen, Xuan-Truong Hoang, Sébastien Destercke, Cassio de Campos,
> Van-Nam Huynh. Vu-Linh Nguyen and Xuan-Truong Hoang contributed equally to
> this work and should be regarded as co-first authors.
> Accepted for publication in *Machine Learning* (Springer), 2026.
> The DOI will be added once the Version of Record is published online.

> **Abstract.** In this paper, we explore how multi-label classification
> (MLC) tasks can be cast into order structure learning. Our motivation for
> doing so is to exploit the very rich structure of the orders to improve and
> robustify MLC learning. We describe formally how MLC can be transformed into
> an order structure learning and prediction task, and then proceed to study
> the problem of predicting Bayes-optimal order structures. We then perform
> some experiments in settings where the use of order structures can be very
> beneficial: robust MLC in the presence of noisy and imbalanced labels, and
> making MLC predictions with partial abstention.
>
> **Keywords:** MLC; preference learning; noisy and imbalanced labels;
> robustness.

## What it does

The method is built around **Bayes-Optimal Preference Orders (BOPOs)**.
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
@article{nguyen2026robust,
  title     = {Robust multi-label classification via preference learning},
  author    = {Nguyen, Vu-Linh and Hoang, Xuan-Truong and Destercke, S{\'e}bastien and de Campos, Cassio and Huynh, Van-Nam},
  journal   = {Machine Learning},
  publisher = {Springer},
  year      = {2026},
}
```

## License

Released under the [MIT License](LICENSE).
