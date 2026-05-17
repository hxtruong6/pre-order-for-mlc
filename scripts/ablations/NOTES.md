# Ablation & performance notes

Reference notes from ad-hoc benchmarks. Not part of the paper; kept here so
future runs don't repeat the same investigations.

## LightGBM thread oversubscription (fixed 2026-05-18)

`preorder4mlc.estimator.Estimator.get_classifier` previously constructed
`LGBMClassifier(n_jobs=number_of_cores - 1)`. Every call site for LGBM in
the pipeline wraps it in `joblib.Parallel(n_jobs=-1)`
(`base_classifiers.py` for all three pairwise variants; ECC/LP path in
`train_extra_baselines.py`). The product was `cores × (cores − 1)` competing
threads — severe oversubscription on machines with ≥8 cores.

Fix: pin `n_jobs=1` in the LightGBM constructor so the outer joblib loop
owns parallelism. Single source of truth.

### Bench: yeast, K=14, 91 pairs (3 repeats, mean ± std)

Fair comparison — each variant uses single-thread per-fit + joblib n_jobs=-1
outside. Script: `scripts/ablations/bench_rf_vs_lgbm.py`.

| Base learner          | Mean   | vs RF                       |
|-----------------------|--------|-----------------------------|
| LightGBM (n_jobs=1)   | 13.6 s | 0.55× (1.8× faster than RF) |
| RF                    | 24.9 s | 1.00×                       |
| HGB                   | 37.4 s | 1.50× (slower than RF)      |

Before the fix, LightGBM was ~72 s on the same dataset (5.3× slower than
its true cost) due to oversubscription.

### Implications

- `--base_learner lgbm` is now genuinely faster than RF for the paper's
  small-to-medium-K datasets. Worth re-running
  `scripts/ablations/ablation_base_learner.py` if any LGBM wall-time
  numbers were quoted previously.
- Default base learner remains RF (`config.py::BASE_LEARNERS`) to preserve
  reproducibility of paper tables. LGBM stays opt-in.
- `HistGradientBoostingClassifier` was tested and lost to both — not worth
  adding to the codebase.

To reproduce:

```bash
PYTHONPATH=. python scripts/ablations/bench_rf_vs_lgbm.py \
    --dataset yeast --repeats 3
```
