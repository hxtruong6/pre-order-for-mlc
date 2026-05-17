"""Smoke run for large-K datasets — single train/test split, time each stage.

Goal: measure wall-clock for fit / predict_proba / preference-order extraction
on one large-K dataset (mediamill, CAL500, bibtex) without burning a full
K-fold CV. Use this to decide whether a full benchmark is feasible on the
current machine.

Example:
    PYTHONPATH=. python scripts/smoke_large_k.py \\
        --dataset mediamill --order PR --train_frac 0.8
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from preorder4mlc.config import ConfigManager
from preorder4mlc.constants import BaseLearnerName, TargetMetric
from preorder4mlc.datasets4experiments import Datasets4Experiments
from preorder4mlc.evaluation_metric import EvaluationMetric
from preorder4mlc.inference_models import PredictBOPOs, PreferenceOrder

ROOT = Path(__file__).resolve().parent.parent


def _rss_mb() -> float:
    import resource
    # ru_maxrss on macOS is in bytes, on linux it's in kB
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def _stage(label: str, fn):
    print(f"[{_rss_mb():>7.0f} MB] {label}...", flush=True)
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    print(f"[{_rss_mb():>7.0f} MB] {label}: {dt / 60:.2f} min ({dt:.1f}s)", flush=True)
    return out, dt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="water_quality")
    p.add_argument("--order", default="PR", choices=["PR", "PA"])
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--base_learner", default=BaseLearnerName.RF.value)
    args = p.parse_args()

    ds_cfg = ConfigManager.get_dataset_config(args.dataset)
    loader = Datasets4Experiments(
        data_path=str(ROOT / "data") + "/",
        data_files=[{"dataset_name": ds_cfg.file, "n_labels_set": ds_cfg.n_labels}],
    )
    print(f"=== {args.dataset}  order={args.order}  base={args.base_learner} ===", flush=True)
    _, _ = _stage("load dataset", loader.load_datasets)
    X, Y, name = loader.datasets[0]
    n_labels = Y.shape[1]
    print(f"  shapes: X{X.shape} Y{Y.shape}  K={n_labels}  pairs={n_labels * (n_labels - 1) // 2}", flush=True)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(X))
    n_tr = int(args.train_frac * len(X))
    X_tr, X_te = X[perm[:n_tr]], X[perm[n_tr:]]
    Y_tr, Y_te = Y[perm[:n_tr]], Y[perm[n_tr:]]
    print(f"  split: train={len(X_tr)}  test={len(X_te)}", flush=True)

    po = PreferenceOrder.PRE_ORDER if args.order == "PR" else PreferenceOrder.PARTIAL_ORDER

    m = PredictBOPOs(args.base_learner, preference_order=po)
    _, t_fit = _stage("fit", lambda: m.fit(X_tr, Y_tr))
    gc.collect()

    proba, t_pp = _stage("predict_proba", lambda: m.predict_proba(X_te, n_labels))

    (pred_bv, _pred_order, _, _), t_inf = _stage(
        "predict_preference_orders (Hamming target)",
        lambda: m.predict_preference_orders(proba, n_labels, len(X_te), TargetMetric.Hamming, height=None),
    )

    pred_bv = np.asarray(pred_bv, dtype=int)
    em = EvaluationMetric()
    print("\n=== metrics ===", flush=True)
    print(f"  f1                : {em.f1(pred_bv, Y_te):.4f}")
    print(f"  hamming_accuracy  : {em.hamming_accuracy(pred_bv, Y_te):.4f}")
    print(f"  subset0_1         : {em.subset0_1(pred_bv, Y_te):.4f}")

    print("\n=== timing summary ===", flush=True)
    print(f"  fit                            : {t_fit / 60:.2f} min")
    print(f"  predict_proba                  : {t_pp / 60:.2f} min")
    print(f"  predict_preference_orders      : {t_inf / 60:.2f} min")
    print(f"  peak RSS                       : {_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
