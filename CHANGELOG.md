# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-08-26

### Changed
- Align the README and publication metadata with the accepted paper.
- Record Vu-Linh Nguyen and Xuan-Truong Hoang as co-first authors and use the
  paper's author order consistently.
- Defer the DOI entry until the Version of Record is published online.

## [1.0.0] - 2026-05-15

First public release accompanying the paper *"Robust multi-label classification via
preference learning"* (Machine Learning, Springer, 2026).

### Added
- Order-structure formulation of MLC: pairwise preference classifiers
  (`base_classifiers.py`), inference models (`inference_models.py`), and ILP-based
  search (`searching_algorithms.py`, `solvers.py`) using HiGHS / CVXOPT.
- Bayes-optimal (pre)order prediction with support for partial abstention and a
  cost-sensitive layer (`cost_sensitive.py`).
- Evaluation metrics (`evaluation_metric.py`), training orchestrator
  (`training_orchestrator.py`), and dataset loaders (`datasets4experiments.py`).
- Command-line training/evaluation drivers under `scripts/` and an end-to-end
  reproduction driver `run.sh`.
- Result aggregation, statistical tests, and figure generation under
  `preorder4mlc/utils/`.
- Packaging and release infrastructure: PyPI metadata, Zenodo deposition metadata,
  CI and publish workflows, Docker reproducibility capsule.
