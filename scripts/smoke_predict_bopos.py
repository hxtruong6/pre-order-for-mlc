"""Behavior-preserving smoke test for PredictBOPOs.

Runs every public method of PredictBOPOs on one train/test split of
chd_49 and pickles the outputs to ``<out_dir>/smoke.pkl``. Run before
and after the refactor and diff the pickles.
"""

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np

from preorder4mlc.constants import BaseLearnerName, TargetMetric
from preorder4mlc.datasets4experiments import Datasets4Experiments
from preorder4mlc.inference_models import PredictBOPOs, PreferenceOrder

ROOT = Path(__file__).resolve().parent.parent


def _hash(obj) -> str:
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = Datasets4Experiments(
        data_path=str(ROOT / "data") + "/",
        data_files=[{"dataset_name": "CHD_49.arff", "n_labels_set": 6}],
    )
    loader.load_datasets()
    X, Y, _ = loader.get_datasets()[0]
    n_labels = Y.shape[1]

    # One fixed split, no noise.
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(X))
    cut = int(len(X) * 0.7)
    train_idx, test_idx = perm[:cut], perm[cut:]
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train = Y[train_idx]

    outputs = {}

    for po in [PreferenceOrder.PRE_ORDER, PreferenceOrder.PARTIAL_ORDER]:
        m = PredictBOPOs(BaseLearnerName.RF.value, preference_order=po)
        m.fit(X_train, Y_train)
        proba = m.predict_proba(X_test, n_labels)
        outputs[f"predict_proba_{po.name}"] = proba

        for metric in [TargetMetric.Hamming, TargetMetric.Subset]:
            for h in [None, 2]:
                pred_bopos, pred_bv, idx_vec, pred_pa = m.predict_preference_orders(
                    proba, n_labels, len(X_test), metric, height=h
                )
                outputs[f"po_{po.name}_{metric.name}_h{h}"] = {
                    "predict_BOPOS": pred_bopos,
                    "predict_binary_vectors": pred_bv,
                    "prediction_with_partial_abstention": pred_pa,
                }

        marginal = m.predict_marginal_proba(X_test)
        outputs[f"marginal_{po.name}"] = marginal

    # CLR path.
    m_clr = PredictBOPOs(BaseLearnerName.RF.value, preference_order=PreferenceOrder.PRE_ORDER)
    m_clr.fit_CLR(X_train, Y_train)
    clr_y, clr_ranks, clr_proba = m_clr.predict_CLR(X_test, n_labels)
    outputs["predict_CLR"] = {"Y": clr_y, "ranks": clr_ranks, "proba": clr_proba}

    # BR baseline.
    m_br = PredictBOPOs(BaseLearnerName.RF.value, preference_order=PreferenceOrder.PRE_ORDER)
    m_br.fit_BR(X_train, Y_train)
    br_y, br_ranks, br_proba = m_br.predict_BR(X_test, n_labels)
    outputs["predict_BR"] = {"Y": br_y, "ranks": br_ranks, "proba": br_proba}

    # CC baseline.
    m_cc = PredictBOPOs(BaseLearnerName.RF.value, preference_order=PreferenceOrder.PRE_ORDER)
    m_cc.fit_CC(X_train, Y_train)
    cc_y, cc_ranks, cc_proba = m_cc.predict_CC(X_test, n_labels)
    outputs["predict_CC"] = {"Y": cc_y, "ranks": cc_ranks, "proba": cc_proba}

    pkl_path = out_dir / "smoke.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(outputs, f)

    digest = _hash(outputs)
    (out_dir / "smoke.sha256").write_text(digest + "\n")
    print(f"wrote {pkl_path}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
