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
import pickle
import time
from logging import INFO, basicConfig, log
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

# Monkey-patch MLkNN._compute_cond for the modern sklearn API used in env.
import skmultilearn.adapt.mlknn as _mlknn_mod
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
N_REPEAT = 2
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
) -> np.ndarray:
    """Ensemble of Classifier Chains with random orders + bagging-style sampling."""
    rng = np.random.RandomState(rng_seed)
    n_labels = Y_train.shape[1]
    n_train = X_train.shape[0]

    votes = np.zeros((X_test.shape[0], n_labels), dtype=np.float64)

    for k in range(n_ensembles):
        perm = rng.permutation(n_labels)
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

        inv_perm = np.argsort(perm)
        pred = pred[:, inv_perm]
        votes += pred

    return (votes >= (n_ensembles / 2.0)).astype(int)


def train_one(
    algo: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    if algo == "mlknn":
        clf = MLkNN(k=10)
        clf.fit(X_train, Y_train)
        return _to_dense_int(clf.predict(X_test))

    if algo == "lp":
        clf = LabelPowerset(
            classifier=RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            ),
            require_dense=[True, True],
        )
        clf.fit(X_train, Y_train)
        return _to_dense_int(clf.predict(X_test))

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
                Y_pred = train_one(algo, X_train, Y_train, X_test)
                log(
                    INFO,
                    f"fold={fold+1} time={(time.time()-t0):.2f}s "
                    f"Y_test={Y_test.shape} Y_pred={Y_pred.shape}",
                )

                record = {
                    "Y_test": Y_test.tolist(),
                    "Y_predicted": Y_pred.tolist(),
                    "Y_BOPOs": [],
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
            Path(results_dir)
            / f"dataset_{dataset_cfg.name.lower()}_noisy_{noisy_rate}_{algo}.pkl"
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
