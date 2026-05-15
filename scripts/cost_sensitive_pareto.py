"""Cost-sensitive Hamming Pareto sweep (paper Appendix G).

Trains a Binary Relevance probability model once per fold, then sweeps the
Bayes-optimal threshold τ = c_FP / (c_FP + c_FN) over a geometric range of
cost ratios. Reports F1, Hamming accuracy, and cost-sensitive Hamming
accuracy at each ratio to expose the F1 ↔ Hamming Pareto curve.

Why BR-marginals: cost-sensitive Hamming decomposes per-label, so the
optimal decision only needs marginal P(y_k = 1 | x). BR estimates exactly
that, and one fit is enough to sweep all τ — far cheaper than retraining
PA/PR for every cost ratio.

Example:
    PYTHONPATH=. python scripts/cost_sensitive_pareto.py \\
        --dataset water_quality --folds 5 --noise 0.0 \\
        --start 0.5 --stop 5.0 --n 9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from preorder4mlc.config import ConfigManager
from preorder4mlc.constants import RANDOM_STATE
from preorder4mlc.cost_sensitive import (
    cost_sensitive_threshold,
    pareto_sweep_costs,
    predict_cost_sensitive,
)
from preorder4mlc.datasets4experiments import Datasets4Experiments
from preorder4mlc.evaluation_metric import EvaluationMetric

ROOT = Path(__file__).resolve().parent.parent


def _fit_br_proba(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Return per-label marginal probabilities P(y_k = 1 | x) on X_test."""
    n_labels = Y_train.shape[1]
    proba = np.zeros((X_test.shape[0], n_labels), dtype=float)
    for k in range(n_labels):
        y_k = Y_train[:, k]
        clf = RandomForestClassifier(random_state=RANDOM_STATE)
        if len(np.unique(y_k)) < 2:
            proba[:, k] = float(y_k.mean())
            continue
        clf.fit(X_train, y_k)
        p = clf.predict_proba(X_test)
        pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
        proba[:, k] = p[:, pos_idx]
    return proba


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="water_quality")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--start", type=float, default=0.5)
    p.add_argument("--stop", type=float, default=5.0)
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--out_dir", default="results/cost_sensitive_pareto")
    args = p.parse_args()

    ds_cfg = ConfigManager.get_dataset_config(args.dataset)
    loader = Datasets4Experiments(
        data_path=str(ROOT / "data") + "/",
        data_files=[{"dataset_name": ds_cfg.file, "n_labels_set": ds_cfg.n_labels}],
    )
    loader.load_datasets()
    X, Y, _name = loader.datasets[0]
    if args.noise > 0:
        Y = loader.add_noise_to_labels(Y.copy(), args.noise)

    em = EvaluationMetric()
    cost_pairs = pareto_sweep_costs(args.start, args.stop, args.n)
    print(f"Dataset {args.dataset}: X{X.shape} Y{Y.shape}  sweeping {len(cost_pairs)} cost ratios")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(X))
    folds_idx = np.array_split(perm, args.folds)

    rows = []
    for fold_i in range(args.folds):
        test_idx = folds_idx[fold_i]
        train_idx = np.concatenate([folds_idx[j] for j in range(args.folds) if j != fold_i])
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        print(f"  fold {fold_i + 1}/{args.folds}: fitting BR on {X_tr.shape[0]} × {Y_tr.shape[1]}...")
        proba = _fit_br_proba(X_tr, Y_tr, X_te)

        for cost_fp, cost_fn in cost_pairs:
            thr = cost_sensitive_threshold(cost_fp, cost_fn)
            pred = predict_cost_sensitive(proba, cost_fp, cost_fn)
            rows.append({
                "fold": fold_i,
                "cost_fp": cost_fp,
                "cost_fn": cost_fn,
                "threshold": thr,
                "f1": em.f1(pred, Y_te),
                "hamming_accuracy": em.hamming_accuracy(pred, Y_te),
                "subset0_1": em.subset0_1(pred, Y_te),
                "cs_hamming_accuracy": em.cs_hamming_accuracy(pred, Y_te, cost_fp, cost_fn),
            })

    df = pd.DataFrame(rows)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.dataset}_noise{args.noise}_folds{args.folds}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    agg = df.groupby(["cost_fp", "cost_fn"]).agg(
        threshold=("threshold", "first"),
        f1_mean=("f1", "mean"),
        hamming_mean=("hamming_accuracy", "mean"),
        subset_mean=("subset0_1", "mean"),
        cs_hamming_mean=("cs_hamming_accuracy", "mean"),
    ).reset_index()
    print("\n=== Pareto curve (mean over folds) ===")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(agg.to_string(index=False))

    plain = agg[np.isclose(agg["cost_fp"], 1.0) & np.isclose(agg["cost_fn"], 1.0)]
    if not plain.empty:
        base_f1 = float(plain["f1_mean"].iloc[0])
        base_ham = float(plain["hamming_mean"].iloc[0])
        print(f"\nReference (c_FP=c_FN=1, τ=0.5):  F1={base_f1:.4f}  Hamming={base_ham:.4f}")
        best_f1 = agg.loc[agg["f1_mean"].idxmax()]
        best_ham = agg.loc[agg["hamming_mean"].idxmax()]
        print(f"Best F1:      c=({best_f1['cost_fp']:.2f},{best_f1['cost_fn']:.2f}) τ={best_f1['threshold']:.3f}  F1={best_f1['f1_mean']:.4f}  Hamming={best_f1['hamming_mean']:.4f}")
        print(f"Best Hamming: c=({best_ham['cost_fp']:.2f},{best_ham['cost_fn']:.2f}) τ={best_ham['threshold']:.3f}  F1={best_ham['f1_mean']:.4f}  Hamming={best_ham['hamming_mean']:.4f}")


if __name__ == "__main__":
    main()
