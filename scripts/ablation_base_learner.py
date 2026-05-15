"""A/B/C/D ablation: RF vs LightGBM × calibration off/on.

Runs a *minimal* PA/PR pipeline on one dataset, one noise level, a small
number of folds, for each of the four configurations and reports the
metric deltas. Goal: decide whether to flip the default base learner /
calibration setting in preorder4mlc/config.py and estimator.py.

This intentionally bypasses TrainingOrchestrator + the full evaluate.py
pickle pipeline -- we want a fast comparison, not a full benchmark.

Architecture: each config runs in a *subprocess* so the
``PREORDER_CALIBRATE`` env var is read fresh by ``preorder4mlc.estimator``
at module-load time. Reloading inside one process doesn't propagate to
already-imported dependents (PredictBOPOs etc.).

Example:
    PYTHONPATH=. python scripts/ablation_base_learner.py \\
        --dataset emotions --folds 3 --noise 0.0 \\
        --out_dir results/ablation_lgbm_calib

Cost: one fold of PA on emotions trains ~K*(K-1)/2 pairwise classifiers,
each ~seconds with RF/LightGBM. 3 folds × 4 configs typically completes
in a few minutes on a laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Inner worker mode: invoked by subprocess. Runs one config end-to-end and
# writes a JSON file with the metric results to --worker_out.
WORKER_FLAG = "--__worker"


def _worker(args: argparse.Namespace) -> None:
    """Run one (base_learner, calibrate) config in this fresh interpreter."""
    # PREORDER_CALIBRATE is read at preorder4mlc.estimator import time, so the
    # env var must be set *before* the first import below.
    assert os.environ.get("PREORDER_CALIBRATE") in ("0", "1")

    from preorder4mlc.config import ConfigManager
    from preorder4mlc.constants import TargetMetric
    from preorder4mlc.datasets4experiments import Datasets4Experiments
    from preorder4mlc.evaluation_metric import EvaluationMetric
    from preorder4mlc.inference_models import PredictBOPOs, PreferenceOrder

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
    n_labels = Y.shape[1]
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(X))
    folds_idx = np.array_split(perm, args.folds)

    metric_keys = ["f1", "hamming_accuracy", "subset0_1", "afrd", "mfrd"]
    per_classifier: dict[str, dict[str, list[float]]] = {}

    def _record(tag: str, pred_bv: np.ndarray, Y_test: np.ndarray) -> None:
        row = per_classifier.setdefault(tag, {k: [] for k in metric_keys})
        row["f1"].append(em.f1(pred_bv, Y_test))
        row["hamming_accuracy"].append(em.hamming_accuracy(pred_bv, Y_test))
        row["subset0_1"].append(em.subset0_1(pred_bv, Y_test))
        row["afrd"].append(em.afrd(pred_bv, Y_test))
        row["mfrd"].append(em.mfrd(pred_bv, Y_test))

    for fold_i in range(args.folds):
        test_idx = folds_idx[fold_i]
        train_idx = np.concatenate([folds_idx[j] for j in range(args.folds) if j != fold_i])
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        for po, po_tag in [
            (PreferenceOrder.PRE_ORDER, "PR-H"),
            (PreferenceOrder.PARTIAL_ORDER, "PA-H"),
        ]:
            m = PredictBOPOs(args.base_learner, preference_order=po)
            m.fit(X_train, Y_train)
            proba = m.predict_proba(X_test, n_labels)
            pred_bv, _pred_order, _, _ = m.predict_preference_orders(
                proba, n_labels, len(X_test), TargetMetric.Hamming, height=None
            )
            pred_bv = np.asarray(pred_bv, dtype=int)
            _record(po_tag, pred_bv, Y_test)

        m_br = PredictBOPOs(args.base_learner, preference_order=PreferenceOrder.PRE_ORDER)
        m_br.fit_BR(X_train, Y_train)
        br_y, _, _ = m_br.predict_BR(X_test, n_labels)
        _record("BR", np.asarray(br_y, dtype=int), Y_test)

    out = {"per_classifier": per_classifier}
    Path(args.worker_out).write_text(json.dumps(out))


def _spawn(cfg_name: str, base_learner: str, calibrate: bool, args: argparse.Namespace) -> dict:
    worker_out = Path(args.out_dir) / f"_worker_{cfg_name}.json"
    worker_out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PREORDER_CALIBRATE"] = "1" if calibrate else "0"
    env["PYTHONPATH"] = str(ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        sys.executable,
        __file__,
        WORKER_FLAG,
        "--dataset", args.dataset,
        "--folds", str(args.folds),
        "--noise", str(args.noise),
        "--seed", str(args.seed),
        "--base_learner", base_learner,
        "--worker_out", str(worker_out),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"[{cfg_name}] subprocess failed (rc={proc.returncode})")
        print("--- stderr ---")
        print(proc.stderr[-2000:])
        raise RuntimeError(f"{cfg_name} failed")
    data = json.loads(worker_out.read_text())
    data["_elapsed_s"] = elapsed
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="emotions")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--out_dir", default="results/ablation_base_learner")
    p.add_argument(
        "--configs",
        default="rf_nocalib,rf_calib,lgbm_nocalib,lgbm_calib",
        help="Comma-separated subset of configs.",
    )
    # worker mode flags
    p.add_argument(WORKER_FLAG, action="store_true", dest="worker")
    p.add_argument("--base_learner", default=None)
    p.add_argument("--worker_out", default=None)
    args = p.parse_args()

    if args.worker:
        _worker(args)
        return

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}  folds={args.folds}  noise={args.noise}")

    config_specs = {
        "rf_nocalib": ("RF", False),
        "rf_calib": ("RF", True),
        "lgbm_nocalib": ("LightGBM", False),
        "lgbm_calib": ("LightGBM", True),
    }
    selected = [c.strip() for c in args.configs.split(",")]
    rows = []
    for cfg_name in selected:
        base_learner, calibrate = config_specs[cfg_name]
        print(f"\n=== running {cfg_name}  (base={base_learner}, calibrate={calibrate}) ===")
        result = _spawn(cfg_name, base_learner, calibrate, args)
        print(f"  done in {result['_elapsed_s']:.1f}s")
        per_clf = result["per_classifier"]
        for clf, metrics in per_clf.items():
            for m, vals in metrics.items():
                rows.append({
                    "config": cfg_name,
                    "base_learner": base_learner,
                    "calibrate": calibrate,
                    "classifier": clf,
                    "metric": m,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "elapsed_s": result["_elapsed_s"],
                })

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{args.dataset}_noise{args.noise}_folds{args.folds}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    pivot = df.pivot_table(index=["classifier", "metric"], columns="config", values="mean")
    print("\n=== Comparison (mean over folds) ===")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(pivot.to_string())

    higher = {"f1", "hamming_accuracy", "subset0_1"}
    lower = {"afrd", "mfrd"}
    print("\n=== Per-(classifier, metric) winner; Δ vs rf_nocalib ===")
    for (clf, metric), row in pivot.iterrows():
        vals = row.dropna().to_dict()
        if not vals:
            continue
        if metric in higher:
            winner = max(vals, key=vals.get)
        elif metric in lower:
            winner = min(vals, key=vals.get)
        else:
            continue
        base = vals.get("rf_nocalib")
        delta = (vals[winner] - base) if base is not None else float("nan")
        print(f"  {clf:<6} {metric:<18} winner={winner:<14} Δ={delta:+.4f}")


if __name__ == "__main__":
    main()
