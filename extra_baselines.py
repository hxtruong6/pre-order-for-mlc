"""Train MLkNN, ECC, and LP baselines.

Standalone, additive script. Loads datasets via Datasets4Experiments using the
same splits / seed as main.py so the new baseline results are directly
comparable to existing BR/CC/CLR/BOPOs pickles.

CLI:
    python extra_baselines.py --dataset <key> --results_dir <dir> --algorithm <mlknn|ecc|lp>

Output:
    results/<dir>/dataset_<name>_noisy_<rate>_<algo>.pkl
"""

import argparse
import logging
import pickle
import time
from logging import INFO, basicConfig, log
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

# Monkey-patch MLkNN._compute_cond for the modern sklearn API used in env.
import skmultilearn.adapt.mlknn as _mlknn_mod
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from skmultilearn.utils import get_matrix_in_format


def _patched_compute_cond(self, X, y):
    self.knn_ = NearestNeighbors(n_neighbors=self.k).fit(X)
    c = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="i8")
    cn = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="i8")
    label_info = get_matrix_in_format(y, "dok")
    neighbors = [
        a[self.ignore_first_neighbours :]
        for a in self.knn_.kneighbors(
            X, self.k + self.ignore_first_neighbours, return_distance=False
        )
    ]
    for instance in range(self._num_instances):
        deltas = label_info[neighbors[instance], :].sum(axis=0)
        for label in range(self._num_labels):
            if label_info[instance, label] == 1:
                c[label, deltas[0, label]] += 1
            else:
                cn[label, deltas[0, label]] += 1
    c_sum = c.sum(axis=1)
    cn_sum = cn.sum(axis=1)
    cond_prob_true = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="float")
    cond_prob_false = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="float")
    for label in range(self._num_labels):
        for neighbor in range(self.k + 1):
            cond_prob_true[label, neighbor] = (self.s + c[label, neighbor]) / (
                self.s * (self.k + 1) + c_sum[label, 0]
            )
            cond_prob_false[label, neighbor] = (self.s + cn[label, neighbor]) / (
                self.s * (self.k + 1) + cn_sum[label, 0]
            )
    return cond_prob_true, cond_prob_false


_mlknn_mod.MLkNN._compute_cond = _patched_compute_cond

from skmultilearn.adapt import MLkNN  # noqa: E402
from skmultilearn.problem_transform import ClassifierChain, LabelPowerset  # noqa: E402

from config import ConfigManager  # noqa: E402
from constants import RANDOM_STATE  # noqa: E402
from datasets4experiments import Datasets4Experiments  # noqa: E402

NOISY_RATES = [0.0, 0.1, 0.2, 0.3]
N_REPEAT = 5
N_FOLDS = 5
BASE_LEARNER = "RF"


def _to_dense_int(M) -> np.ndarray:
    if sparse.issparse(M):
        M = M.toarray()
    return np.asarray(M).astype(int)


def _ecc_predict(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    n_ensembles: int = 10,
    rng_seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble of Classifier Chains with random orders + bagging-style sampling.

    Returns:
        Y_pred: (n_test, n_labels) binary majority-vote prediction.
        Y_proba: (n_test, n_labels) mean per-label probability across chains
                 in [0, 1].
    """
    rng = np.random.RandomState(rng_seed)
    n_labels = Y_train.shape[1]
    n_train = X_train.shape[0]

    votes = np.zeros((X_test.shape[0], n_labels), dtype=np.float64)
    proba_sum = np.zeros((X_test.shape[0], n_labels), dtype=np.float64)

    for k in range(n_ensembles):
        perm = rng.permutation(n_labels)
        inv_perm = np.argsort(perm)
        # bootstrap sample
        idx = rng.randint(0, n_train, size=n_train)
        X_bs = X_train[idx]
        Y_bs = Y_train[idx][:, perm]

        chain = ClassifierChain(
            classifier=RandomForestClassifier(
                n_estimators=100, random_state=rng_seed + k, n_jobs=-1
            ),
            require_dense=[True, True],
        )
        chain.fit(X_bs, Y_bs)
        pred = chain.predict(X_test)
        pred = _to_dense_int(pred)

        pred = pred[:, inv_perm]
        votes += pred

        # Per-chain marginal proba in the chain's (permuted) label order.
        try:
            proba = chain.predict_proba(X_test)
            if sparse.issparse(proba):
                proba = proba.toarray()
            proba = np.asarray(proba, dtype=float)
            # Invert permutation so columns line up with original label order.
            proba = proba[:, inv_perm]
            proba_sum += proba
        except Exception as e:  # pragma: no cover - defensive
            logging.warning("ECC chain %d predict_proba failed: %s", k, e)
            # Fall back to using the hard prediction as a degenerate "proba".
            proba_sum += pred

    Y_pred = (votes >= (n_ensembles / 2.0)).astype(int)
    Y_proba = np.clip(proba_sum / float(n_ensembles), 0.0, 1.0)
    return Y_pred, Y_proba


def _lp_marginal_proba(clf: "LabelPowerset", X: np.ndarray, n_labels: int) -> np.ndarray:
    """Per-label marginal probabilities from a LabelPowerset.

    skmultilearn (0.2.x) returns *per-label* marginals directly from
    LabelPowerset.predict_proba (shape (n, n_labels)) -- it internally
    aggregates the meta-class probabilities for us. We just densify and
    clip to [0, 1].

    Returns (n, n_labels) float. Returns None on unexpected shapes so the
    caller can fall back to Y_pred.astype(float).
    """
    proba = clf.predict_proba(X)
    if sparse.issparse(proba):
        proba = proba.toarray()
    proba = np.asarray(proba, dtype=float)

    if proba.ndim != 2 or proba.shape[1] != n_labels:
        # If the underlying skmultilearn returns meta-class probabilities
        # instead of per-label marginals, fall back to a manual aggregation
        # using reverse_combinations_ (a list-of-lists of positive label
        # indices, one entry per meta-class).
        reverse = getattr(clf, "reverse_combinations_", None)
        if reverse is None:
            return None  # type: ignore[return-value]
        n_meta = proba.shape[1]
        if len(reverse) != n_meta:
            return None  # type: ignore[return-value]
        bit_mat = np.zeros((n_meta, n_labels), dtype=float)
        for m, pos_indices in enumerate(reverse):
            for k in pos_indices:
                if 0 <= k < n_labels:
                    bit_mat[m, k] = 1.0
                else:
                    return None  # type: ignore[return-value]
        proba = proba @ bit_mat

    return np.clip(proba, 0.0, 1.0)


def train_one(
    algo: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Train one baseline and return (Y_pred, Y_proba).

    Y_pred is (n_test, n_labels) int array; Y_proba is (n_test, n_labels)
    float array in [0, 1] giving per-label marginal probability.
    """
    n_labels = Y_train.shape[1]

    if algo == "mlknn":
        clf = MLkNN(k=10)
        clf.fit(X_train, Y_train)
        Y_pred = _to_dense_int(clf.predict(X_test))
        try:
            proba = clf.predict_proba(X_test)
            if sparse.issparse(proba):
                proba = proba.toarray()
            Y_proba = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
        except Exception as e:
            logging.warning("MLkNN predict_proba failed: %s", e)
            Y_proba = Y_pred.astype(float)
        return Y_pred, Y_proba

    if algo == "lp":
        clf = LabelPowerset(
            classifier=RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            ),
            require_dense=[True, True],
        )
        clf.fit(X_train, Y_train)
        Y_pred = _to_dense_int(clf.predict(X_test))
        try:
            Y_proba = _lp_marginal_proba(clf, X_test, n_labels)
        except Exception as e:
            logging.warning("LP marginal proba failed: %s", e)
            Y_proba = None
        if Y_proba is None:
            logging.warning(
                "LabelPowerset.unique_combinations_ unavailable or malformed; "
                "falling back to Y_pred as degenerate proba."
            )
            Y_proba = Y_pred.astype(float)
        return Y_pred, Y_proba

    if algo == "ecc":
        return _ecc_predict(X_train, Y_train, X_test, n_ensembles=10)

    raise ValueError(f"Unknown algorithm: {algo}")


def run(dataset_key: str, results_dir: str, algo: str) -> None:
    basicConfig(level=INFO)

    dataset_cfg = ConfigManager.get_dataset_config(dataset_key)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    exp = Datasets4Experiments(
        "./data/",
        [{"dataset_name": dataset_cfg.file, "n_labels_set": dataset_cfg.n_labels}],
    )
    exp.load_datasets()

    for noisy_rate in NOISY_RATES:
        log(INFO, f"=== {dataset_cfg.name} | {algo} | noisy_rate={noisy_rate} ===")
        results = []

        for repeat_time in range(N_REPEAT):
            log(INFO, f"Repeat {repeat_time+1}/{N_REPEAT}")
            for fold, (X_train, Y_train, X_test, Y_test) in enumerate(
                exp.kfold_split_with_noise(
                    dataset_index=0,
                    n_splits=N_FOLDS,
                    noisy_rate=noisy_rate,
                    random_state=RANDOM_STATE,
                )
            ):
                t0 = time.time()
                Y_pred, Y_proba = train_one(algo, X_train, Y_train, X_test)
                log(
                    INFO,
                    f"fold={fold+1} time={(time.time()-t0):.2f}s "
                    f"Y_test={Y_test.shape} Y_pred={Y_pred.shape} "
                    f"Y_proba={Y_proba.shape}",
                )

                record = {
                    "Y_test": Y_test.tolist(),
                    "Y_predicted": Y_pred.tolist(),
                    "Y_BOPOs": [],
                    "Y_proba": Y_proba.tolist(),
                    "indices_vector": None,
                    "partial_abstention": None,
                    "target_metric": None,
                    "preference_order": None,
                    "height": None,
                    "repeat_time": repeat_time,
                    "fold": fold,
                    "dataset_name": dataset_cfg.name,
                    "base_learner_name": BASE_LEARNER,
                    "noisy_rate": noisy_rate,
                }
                results.append(record)

        out = (
            Path(results_dir) / f"dataset_{dataset_cfg.name.lower()}_noisy_{noisy_rate}_{algo}.pkl"
        )
        with open(out, "wb") as f:
            pickle.dump(results, f)
        log(INFO, f"Saved {out} ({len(results)} records)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--algorithm", required=True, choices=["mlknn", "ecc", "lp"])
    args = p.parse_args()
    run(args.dataset, args.results_dir, args.algorithm)


if __name__ == "__main__":
    main()
