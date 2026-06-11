# Conventions

- **Package layout:** flat `preorder4mlc/` package; CLIs live in `scripts/`.
- **Datasets:** registered in `preorder4mlc/config.py` (`ConfigManager.DATASET_CONFIGS`);
  loaded by `preorder4mlc/datasets4experiments.py`.
- **Results:** per-fold CSVs under `results/<run>/`; aggregated by
  `preorder4mlc/utils/summarize_metrics.py`.
- **Solvers:** ILP search uses HiGHS (`highspy`) with a CVXOPT fallback (`solvers.py`).
- **Versioning:** keep `pyproject.toml`, `.zenodo.json`, `CITATION.cff` in lockstep.
