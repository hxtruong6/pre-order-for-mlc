# preorder4MLC: Pre-Order Based Multi-Label Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Code and full experimental results for the paper

> **\<TODO: paper title\>**
> Hoàng Xuân Trường, Vu-Linh Nguyen.
> *\<TODO: journal name\>*, 2026.
> \<TODO: DOI link once assigned\>

## Overview

This repository implements **B**ipartite **O**rdered **P**reference
**O**rders (BOPOs) for multi-label classification (MLC). For each pair
of labels, a probabilistic pairwise classifier is trained; an integer
linear program then searches per instance for the preference order
(either a pre-order or a partial-order) that minimises an expected loss
(Hamming or Subset 0/1). The search admits an optional height
constraint, yielding eight inference algorithms IA1–IA8 plus three
prediction types — `BinaryVector`, `PreferenceOrder`, and
`PartialAbstention` — evaluated against six published baselines
(CLR, BR, CC, MLkNN, ECC, LP).

## Method at a glance

```
              ┌─────────────────────┐
   training → │ K(K-1)/2 pairwise   │ → pairwise probabilities p_ij
              │ calibrated classif. │      (4 classes / pair for pre-order;
              └─────────────────────┘       3 for partial-order)
                         │
                         ▼
              ┌─────────────────────┐
              │ per-instance ILP    │ → preference order
              │ search (cvxopt+GLPK)│      (height ∈ {2, None})
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ derive prediction:  │ → BinaryVector
              │ binary / order /    │   PreferenceOrder
              │ partial abstention  │   PartialAbstention
              └─────────────────────┘
```

A separately trained binary-relevance head produces per-label marginal
probabilities used for the ranking metrics (`ranking_loss`, `one_error`,
`coverage`, `lr_ap`, `auc_macro`, `auc_micro`).

## Repository layout

```
.
├── preorder4mlc/                       # Library package
│   ├── config.py, constants.py         # Run configuration + global seed
│   ├── datasets4experiments.py         # ARFF loading, k-fold splits, label-noise
│   ├── estimator.py                    # Uniform interface over RF / ETC / XGBoost / LightGBM
│   ├── base_classifiers.py             # Pairwise / calibrated classifier factory
│   ├── inference_models.py             # PredictBOPOs (BOPOs + CLR / BR / CC baselines)
│   ├── searching_algorithms.py         # ILP search for pre- and partial-orders
│   ├── training_orchestrator.py        # Training loop over learners × folds × algorithms
│   ├── evaluation_metric.py            # Example-, label-, ranking-, abstention-metrics
│   └── utils/
│       ├── results_manager.py          # Pickle I/O
│       ├── summarize_metrics.py        # Per-dataset summary tables
│       ├── statistical_tests.py        # Friedman + Nemenyi + CD diagrams
│       ├── plot_figures.py             # Paper figure suite
│       └── suppress.py                 # Mute GLPK stdout/stderr
├── scripts/                            # CLI entry points
│   ├── train.py                        # Train BOPOs + CLR / BR / CC
│   ├── evaluate.py                     # Evaluate BOPOs / CLR / BR / CC pickles
│   ├── train_extra_baselines.py        # Train MLkNN / ECC / LP
│   ├── evaluate_extra_baselines.py     # Evaluate MLkNN / ECC / LP pickles
│   └── smoke_predict_bopos.py          # Behavior-preservation smoke test
├── data/                               # 9 ARFF datasets (see REPRODUCE.md §2)
├── results/                            # Per-fold CSVs + aggregated tables (gitignored except CSV/XLSX)
├── docs/                               # Long-form notes
├── run.sh                              # End-to-end reproduction driver
├── REPRODUCE.md                        # Step-by-step reproduction recipe
├── CITATION.cff                        # How to cite this work
├── pyproject.toml                      # Package metadata + lint config
└── requirements.txt                    # Pinned runtime dependencies
```

## Quick start

```bash
# 1. Install (editable so scripts/ can import the package)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Run one dataset end-to-end
python scripts/train.py    --dataset emotions --results_dir results/run-dev
python scripts/evaluate.py --dataset emotions --results_dir results/run-dev

# 3. Reproduce every result in the paper
RESULTS_DIR=results/run-$(date +%Y%m%d) bash run.sh
python -m preorder4mlc.utils.summarize_metrics \
    --results_dir   "${RESULTS_DIR}" \
    --output_dir    "${RESULTS_DIR}_summary"
python -m preorder4mlc.utils.statistical_tests \
    --results_dir   "${RESULTS_DIR}_summary" \
    --output_dir    "${RESULTS_DIR}_summary/stats"
python -m preorder4mlc.utils.plot_figures \
    --results_dir   "${RESULTS_DIR}_summary" \
    --raw_results_dir "${RESULTS_DIR}"
```

See [REPRODUCE.md](REPRODUCE.md) for the full recipe, expected wall
times, and the file layout produced by each step.

## Datasets

Nine multi-label datasets are tracked under `data/` so the pipeline can
run end-to-end after `pip install`:

`chd_49`, `emotions`, `scene`, `yeast`, `water_quality`,
`viruspseaac`, `humanpseaac`, `gpositivepseaac`, `plantpseaac`.

CLI keys match `preorder4mlc.config::ConfigManager.DATASET_CONFIGS`.

## Partial-abstention metrics

A partial-abstention prediction is a vector in `{0, 1, -1}^K` where
`-1` denotes "abstain". The accompanying metrics are defined as:

```
AREC(ŷ, y) = (1/K) · Σ_k 1[ y_k  ∈ ŷ_k ]   where -1 stands for {0, 1}
AABS(ŷ)    = (1/K) · Σ_k 1[ ŷ_k =  -1 ]
REC(ŷ, y)  = 1 if AREC(ŷ, y) = 1 else 0
ABS(ŷ)     = K · AABS(ŷ)
```

Example:

```
ŷ = [1, 0, 1, -1, 0, -1]      y = [0, 0, 1, 1, 0, 0]

AREC = (0 + 1 + 1 + 1 + 1 + 1) / 6 = 4/6
AABS = 2 / 6
```

## Citation

If you use this code, please cite the paper (see [CITATION.cff](CITATION.cff)
once the DOI is assigned):

```bibtex
@article{TODO,
  title   = {<TODO: paper title>},
  author  = {Hoàng, Xuân Trường and Nguyen, Vu-Linh},
  journal = {<TODO: journal name>},
  year    = {2026},
}
```

## License

Released under the [MIT License](LICENSE).
