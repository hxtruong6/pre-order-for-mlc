"""Time pairwise pre-order training: RF vs LightGBM vs HGB on one dataset.

Self-contained: builds the same pair datasets as BaseClassifiers.pairwise_pre_order_classifier_fit
and times joblib-parallel training across three base learners.

Usage:
    python scripts/ablations/bench_rf_vs_lgbm.py --dataset yeast [--repeats 3]
"""

import argparse
import time

import numpy as np
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from preorder4mlc.config import ConfigManager
from preorder4mlc.constants import RANDOM_STATE
from preorder4mlc.datasets4experiments import Datasets4Experiments


def build_pair_datasets(X, Y):
    n_instances, n_labels = Y.shape
    pairs = {}
    for i in range(n_labels - 1):
        for j in range(i + 1, n_labels):
            yi, yj = Y[:, i], Y[:, j]
            lab = np.full(n_instances, -1, dtype=np.int8)
            lab[(yi == 1) & (yj == 0)] = 0
            lab[(yi == 0) & (yj == 1)] = 1
            lab[(yi == 0) & (yj == 0)] = 2
            lab[(yi == 1) & (yj == 1)] = 3
            pairs[f"{i}_{j}"] = (X, lab.astype(int))
    return pairs


def make_estimator(kind: str):
    if kind == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE)
    if kind == "lgbm":
        return LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
            num_leaves=20,
            max_depth=6,
            bagging_fraction=0.9,
            feature_fraction=0.8,
            learning_rate=0.1,
            n_estimators=100,
            min_child_samples=5,
            min_child_weight=0.0001,
            min_split_gain=0.01,
            is_unbalance=True,
            device="cpu",
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    raise ValueError(kind)


def _fit_one(kind, X, y):
    make_estimator(kind).fit(X, y)


def time_fit(kind: str, pairs: dict, repeats: int) -> list[float]:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        Parallel(n_jobs=-1)(
            delayed(_fit_one)(kind, X_p, y_p) for X_p, y_p in pairs.values()
        )
        times.append(time.perf_counter() - t0)
    return times


def summarize(name: str, ts: list[float]) -> None:
    arr = np.array(ts)
    print(f"  {name:10s}  mean={arr.mean():7.2f}s  std={arr.std():5.2f}s  runs={[f'{t:.2f}' for t in ts]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="yeast")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--data_path", default="./data/")
    args = p.parse_args()

    cfg = ConfigManager.get_dataset_config(args.dataset)
    loader = Datasets4Experiments(
        args.data_path, [{"dataset_name": cfg.file, "n_labels_set": cfg.n_labels}]
    )
    loader.load_datasets()
    X, Y, _ = loader.get_datasets()[0]
    n_pairs = cfg.n_labels * (cfg.n_labels - 1) // 2
    print(f"Dataset={cfg.name}  N={X.shape[0]}  D={X.shape[1]}  K={cfg.n_labels}  pairs={n_pairs}")
    print(f"Repeats per learner: {args.repeats}\n")

    pairs = build_pair_datasets(X, Y)

    print("Warming up worker pool…")
    Parallel(n_jobs=-1)(delayed(_fit_one)("rf", X[:64], Y[:64, 0]) for _ in range(4))

    print("\nTiming pairwise training (joblib n_jobs=-1):")
    rf_t = time_fit("rf", pairs, args.repeats)
    summarize("RF", rf_t)
    lgbm_t = time_fit("lgbm", pairs, args.repeats)
    summarize("LightGBM", lgbm_t)
    hgb_t = time_fit("hgb", pairs, args.repeats)
    summarize("HGB", hgb_t)

    rf_mean = np.mean(rf_t)
    print("\nvs RF (lower is faster):")
    print(f"  LGBM/RF = {np.mean(lgbm_t) / rf_mean:.2f}x")
    print(f"  HGB/RF  = {np.mean(hgb_t) / rf_mean:.2f}x")


if __name__ == "__main__":
    main()
