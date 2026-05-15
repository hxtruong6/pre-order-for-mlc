"""Evaluate MLkNN / ECC / LP baseline pickles into a CSV matching the
existing baseline (br/cc/clr) CSV schema.

CLI:
    python evaluate_extra_baselines.py --dataset <key> --results_dir <dir> --algorithm <mlknn|ecc|lp>

Reads:    results/<dir>/dataset_<name>_noisy_<rate>_<algo>.pkl
Writes:   results/<dir>/evaluation_<name>_noisy_<rate>_<algo>.csv (+ .xlsx)
"""

import argparse
import pickle
from logging import ERROR, INFO, basicConfig, log
from pathlib import Path

import numpy as np
import pandas as pd

from config import ConfigManager
from evaluation_metric import EvaluationMetric, EvaluationMetricName

NOISY_RATES = [0.0, 0.1, 0.2, 0.3]

BINARY_METRICS = [
    EvaluationMetricName.HAMMING_ACCURACY,
    EvaluationMetricName.SUBSET0_1,
    EvaluationMetricName.F1,
    EvaluationMetricName.JACCARD,
    EvaluationMetricName.MACRO_F1,
    EvaluationMetricName.MICRO_F1,
    EvaluationMetricName.MACRO_PRECISION,
    EvaluationMetricName.MICRO_PRECISION,
    EvaluationMetricName.MACRO_RECALL,
    EvaluationMetricName.MICRO_RECALL,
    EvaluationMetricName.EXAMPLE_PRECISION,
    EvaluationMetricName.EXAMPLE_RECALL,
    EvaluationMetricName.MFRD,
    EvaluationMetricName.AFRD,
]

RANKING_METRICS = [
    EvaluationMetricName.RANKING_LOSS,
    EvaluationMetricName.ONE_ERROR,
    EvaluationMetricName.COVERAGE,
    EvaluationMetricName.LR_AP,
    EvaluationMetricName.AUC_MACRO,
    EvaluationMetricName.AUC_MICRO,
    EvaluationMetricName.AUPRC_MACRO,
    EvaluationMetricName.AUPRC_MICRO,
]


def _eval_one(
    metric_name: EvaluationMetricName,
    y_pred: np.ndarray,
    y_test: np.ndarray,
    y_proba: np.ndarray = None,  # type: ignore[assignment]
) -> float:
    em = EvaluationMetric()
    if metric_name == EvaluationMetricName.HAMMING_ACCURACY:
        return em.hamming_accuracy(y_pred, y_test)
    if metric_name == EvaluationMetricName.SUBSET0_1:
        return em.subset0_1(y_pred, y_test)
    if metric_name == EvaluationMetricName.F1:
        return em.f1(y_pred, y_test)
    if metric_name == EvaluationMetricName.JACCARD:
        return em.jaccard(y_pred, y_test)
    if metric_name == EvaluationMetricName.MACRO_F1:
        return em.macro_f1(y_pred, y_test)
    if metric_name == EvaluationMetricName.MICRO_F1:
        return em.micro_f1(y_pred, y_test)
    if metric_name == EvaluationMetricName.MACRO_PRECISION:
        return em.macro_precision(y_pred, y_test)
    if metric_name == EvaluationMetricName.MICRO_PRECISION:
        return em.micro_precision(y_pred, y_test)
    if metric_name == EvaluationMetricName.MACRO_RECALL:
        return em.macro_recall(y_pred, y_test)
    if metric_name == EvaluationMetricName.MICRO_RECALL:
        return em.micro_recall(y_pred, y_test)
    if metric_name == EvaluationMetricName.EXAMPLE_PRECISION:
        return em.example_precision(y_pred, y_test)
    if metric_name == EvaluationMetricName.EXAMPLE_RECALL:
        return em.example_recall(y_pred, y_test)
    if metric_name == EvaluationMetricName.MFRD:
        return em.mfrd(y_pred, y_test)
    if metric_name == EvaluationMetricName.AFRD:
        return em.afrd(y_pred, y_test)
    # Ranking metrics (operate on the real-valued score vector y_proba).
    if metric_name == EvaluationMetricName.RANKING_LOSS:
        return em.ranking_loss(y_proba, y_test)
    if metric_name == EvaluationMetricName.ONE_ERROR:
        return em.one_error(y_proba, y_test)
    if metric_name == EvaluationMetricName.COVERAGE:
        return em.coverage(y_proba, y_test)
    if metric_name == EvaluationMetricName.LR_AP:
        return em.lr_ap(y_proba, y_test)
    if metric_name == EvaluationMetricName.AUC_MACRO:
        return em.auc_macro(y_proba, y_test)
    if metric_name == EvaluationMetricName.AUC_MICRO:
        return em.auc_micro(y_proba, y_test)
    if metric_name == EvaluationMetricName.AUPRC_MACRO:
        return em.auprc_macro(y_proba, y_test)
    if metric_name == EvaluationMetricName.AUPRC_MICRO:
        return em.auprc_micro(y_proba, y_test)
    raise ValueError(f"Unknown metric: {metric_name}")


def evaluate_pickle(pkl_path: Path) -> pd.DataFrame:
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    df = pd.DataFrame(records)

    rows = []
    has_proba = "Y_proba" in df.columns
    for base_learner in df["base_learner_name"].unique():
        sub = df[df["base_learner_name"] == base_learner]
        for metric_name in BINARY_METRICS:
            per_fold = []
            for _, row in sub.iterrows():
                y_pred = np.asarray(row["Y_predicted"])
                y_test = np.asarray(row["Y_test"])
                try:
                    per_fold.append(_eval_one(metric_name, y_pred, y_test))
                except Exception as e:
                    log(ERROR, f"Metric {metric_name} failed: {e}")
            if not per_fold:
                continue
            arr = np.asarray(per_fold, dtype=float)
            rows.append(
                {
                    "Base_Learner": base_learner,
                    "Algorithm": None,
                    "Algorithm_Metric": None,
                    "Algorithm_Height": None,
                    "Algorithm_Order": None,
                    "Prediction_Type": "BinaryVector",
                    "Metric": metric_name.value,
                    "Mean": float(arr.mean()),
                    "Std": float(arr.std()),
                }
            )

        # Ranking / score-vector metrics. Skip silently if Y_proba absent
        # (backward compat with the v1 pickles).
        if not has_proba:
            continue
        for metric_name in RANKING_METRICS:
            per_fold = []
            for _, row in sub.iterrows():
                if row.get("Y_proba") is None:
                    continue
                y_proba = np.asarray(row["Y_proba"], dtype=float)
                y_test = np.asarray(row["Y_test"])
                try:
                    per_fold.append(
                        _eval_one(metric_name, None, y_test, y_proba=y_proba)  # type: ignore[arg-type]
                    )
                except Exception as e:
                    log(ERROR, f"Ranking metric {metric_name} failed: {e}")
            if not per_fold:
                continue
            arr = np.asarray(per_fold, dtype=float)
            rows.append(
                {
                    "Base_Learner": base_learner,
                    "Algorithm": None,
                    "Algorithm_Metric": None,
                    "Algorithm_Height": None,
                    "Algorithm_Order": None,
                    "Prediction_Type": "ScoreVector",
                    "Metric": metric_name.value,
                    "Mean": float(np.nanmean(arr)),
                    "Std": float(np.nanstd(arr)),
                }
            )
    return pd.DataFrame(rows)


def run(dataset_key: str, results_dir: str, algo: str) -> None:
    basicConfig(level=INFO)
    cfg = ConfigManager.get_dataset_config(dataset_key)
    out_dir = Path(results_dir)

    for noisy_rate in NOISY_RATES:
        pkl = out_dir / f"dataset_{cfg.name.lower()}_noisy_{noisy_rate}_{algo}.pkl"
        if not pkl.exists():
            log(ERROR, f"Missing {pkl}, skipping")
            continue

        log(INFO, f"Evaluating {pkl}")
        df_out = evaluate_pickle(pkl)

        base = out_dir / f"evaluation_{cfg.name}_noisy_{noisy_rate}_{algo}"
        df_out.to_csv(f"{base}.csv", index=False)
        try:
            with pd.ExcelWriter(f"{base}.xlsx") as w:
                df_out.to_excel(w, sheet_name="Overall", index=False, float_format="%.5f")
        except Exception as e:
            log(ERROR, f"Excel write failed: {e}")
        log(INFO, f"Wrote {base}.csv ({len(df_out)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--algorithm", required=True, choices=["mlknn", "ecc", "lp"])
    args = p.parse_args()
    run(args.dataset, args.results_dir, args.algorithm)


if __name__ == "__main__":
    main()
