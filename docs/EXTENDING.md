# Extending preorder4mlc

## Add a dataset
Register it in `preorder4mlc/config.py` (`ConfigManager.DATASET_CONFIGS`) with its label
count and file orientation, and ensure `preorder4mlc/datasets4experiments.py` can load its
ARFF/CSV. Then add its key to the `DATASETS` array in `run.sh`.

## Add a base classifier
Implement it in `preorder4mlc/base_classifiers.py` following the existing pairwise /
label-ranking interface (must expose `fit` / `predict`-style methods consumed by
`training_orchestrator.py`).

## Add an evaluation metric
Add the metric to `preorder4mlc/evaluation_metric.py`; it is then available to the
evaluation scripts and the summary tables.

## Change the ILP search
The search objective and constraints live in `preorder4mlc/searching_algorithms.py` and
`preorder4mlc/solvers.py` (HiGHS via `highspy`, CVXOPT fallback).
