"""Train the ECC (Ensemble of Classifier Chains) baseline.

Standalone, additive script. Loads datasets via Datasets4Experiments using the
same splits / seed as scripts/train.py so the ECC results are directly
comparable to existing BR/CC/CLR/BOPOs pickles.

CLI:
    python scripts/train_ecc.py --dataset <key> --results_dir <dir> --algorithm ecc

Output:
    results/<dir>/dataset_<name>_noisy_<rate>_ecc[_lgbm].pkl
"""

import argparse
import logging
import pickle
import time
from logging import INFO, basicConfig, log
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import ClassifierChain

from preorder4mlc.config import ConfigManager
from preorder4mlc.constants import RANDOM_STATE
from preorder4mlc.datasets4experiments import Datasets4Experiments


def _make_base_learner(name: str, random_state=None, is_unbalance: bool = True):
    """Return a fresh base-learner instance for the ECC wrapper.

    Choices:
        - 'rf'   : RandomForestClassifier with paper-equivalent defaults.
        - 'lgbm' : LGBMClassifier tuned for small/imbalanced multi-label tasks.

    'rf' is the default so this script reproduces the original paper
    baseline numbers byte-for-byte; pass --base_learner lgbm to compare
    PA/PR + calibration against a stronger ECC/LP base.
    """
    if name == "rf":
        # n_estimators=100 (sklearn default since 0.22, made explicit) and
        # n_jobs=-1 match the original ECC call on main exactly so the default
        # --base_learner rf reproduces the paper baseline numbers. LP on main
        # used a bare RandomForestClassifier(); n_jobs=-1 only speeds that
        # path up — RF determinism depends solely on random_state.
        return RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
    if name == "lgbm":
        return LGBMClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            num_leaves=20,
            max_depth=6,
            learning_rate=0.1,
            min_child_samples=5,
            is_unbalance=is_unbalance,
        )
    raise ValueError(f"Unknown base_learner: {name!r} (choose 'rf' or 'lgbm')")


NOISY_RATES = [0.0, 0.1, 0.2, 0.3]
N_REPEAT = 5
N_FOLDS = 5


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
    base_learner: str = "rf",
    is_unbalance: bool = True,
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
            classifier=_make_base_learner(base_learner, random_state=rng_seed + k, is_unbalance=is_unbalance),
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


def train_one(
    algo: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    base_learner: str = "rf",
    is_unbalance: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Train ECC and return (Y_pred, Y_proba).

    Y_pred is (n_test, n_labels) int array; Y_proba is (n_test, n_labels)
    float array in [0, 1] giving per-label marginal probability.
    """
    if algo == "ecc":
        return _ecc_predict(X_train, Y_train, X_test, n_ensembles=10, base_learner=base_learner, is_unbalance=is_unbalance)

    raise ValueError(f"Unknown algorithm: {algo}")


def run(
    dataset_key: str,
    results_dir: str,
    algo: str,
    base_learner: str = "rf",
    noisy_rate: float | None = None,
    repeat: int | None = None,
    fold: int | None = None,
    is_unbalance: bool = True,
) -> None:
    """Run extra baseline.

    Default (no noisy_rate/repeat/fold): full sweep, one pickle per noise level.
    Per-noise mode (noisy_rate only): one pickle for that noise level only.
    Split mode (noisy_rate AND repeat AND fold): run a single (R,F) cell and
    write ``dataset_<name>_noisy_<r>_<algo>[_lgbm]_r<R>_f<F>.pkl`` so the
    existing merge_split_results.py can consolidate them.
    """
    basicConfig(level=INFO)

    dataset_cfg = ConfigManager.get_dataset_config(dataset_key)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    exp = Datasets4Experiments(
        "./data/",
        [{"dataset_name": dataset_cfg.file, "n_labels_set": dataset_cfg.n_labels}],
    )
    exp.load_datasets()

    bl_suffix = f"_{base_learner}" if base_learner != "rf" else ""
    split_mode = noisy_rate is not None and repeat is not None and fold is not None
    if split_mode:
        out = Path(results_dir) / (
            f"dataset_{dataset_cfg.name.lower()}_noisy_{noisy_rate}_{algo}{bl_suffix}"
            f"_r{repeat}_f{fold}.pkl"
        )
        if out.exists():
            log(INFO, f"[skip] {out} already exists")
            return
        rates = [noisy_rate]
    elif noisy_rate is not None:
        rates = [noisy_rate]
    else:
        rates = NOISY_RATES

    for noisy_rate in rates:
        log(INFO, f"=== {dataset_cfg.name} | {algo} | noisy_rate={noisy_rate} ===")
        results = []

        for repeat_time in range(N_REPEAT):
            if split_mode and repeat_time != repeat:
                continue
            log(INFO, f"Repeat {repeat_time+1}/{N_REPEAT}")
            for fold_idx, (X_train, Y_train, X_test, Y_test) in enumerate(
                exp.kfold_split_with_noise(
                    dataset_index=0,
                    n_splits=N_FOLDS,
                    noisy_rate=noisy_rate,
                    random_state=RANDOM_STATE,
                )
            ):
                if split_mode and fold_idx != fold:
                    continue
                t0 = time.time()
                Y_pred, Y_proba = train_one(algo, X_train, Y_train, X_test, base_learner=base_learner, is_unbalance=is_unbalance)
                log(
                    INFO,
                    f"fold={fold_idx+1} time={(time.time()-t0):.2f}s "
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
                    "fold": fold_idx,
                    "dataset_name": dataset_cfg.name,
                    "base_learner_name": base_learner.upper(),
                    "noisy_rate": noisy_rate,
                }
                results.append(record)

        if split_mode:
            out = Path(results_dir) / (
                f"dataset_{dataset_cfg.name.lower()}_noisy_{noisy_rate}_{algo}{bl_suffix}"
                f"_r{repeat}_f{fold}.pkl"
            )
        else:
            out = (
                Path(results_dir) / f"dataset_{dataset_cfg.name.lower()}_noisy_{noisy_rate}_{algo}{bl_suffix}.pkl"
            )
        with open(out, "wb") as f:
            pickle.dump(results, f)
        log(INFO, f"Saved {out} ({len(results)} records)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--algorithm", required=True, choices=["ecc"])
    p.add_argument(
        "--base_learner",
        choices=["rf", "lgbm"],
        default="rf",
        help="Base learner for the ECC chains. Default 'rf' reproduces "
        "the original paper baseline; use 'lgbm' for the LGBM variant.",
    )
    p.add_argument(
        "--noisy_rate", "--noise_rate", dest="noisy_rate", type=float, default=None,
        help="Run only this noise level (split mode requires --repeat and --fold too).",
    )
    p.add_argument("--repeat", type=int, default=None, help="Split mode: 0-indexed repeat.")
    p.add_argument("--fold", type=int, default=None, help="Split mode: 0-indexed fold.")
    p.add_argument(
        "--no_is_unbalance",
        action="store_true",
        default=False,
        help="Disable is_unbalance=True in LGBMClassifier. Useful when clean "
        "labels (noise_rate=0.0) cause extreme class weights and slow training.",
    )
    args = p.parse_args()
    run(
        args.dataset, args.results_dir, args.algorithm,
        base_learner=args.base_learner,
        noisy_rate=args.noisy_rate, repeat=args.repeat, fold=args.fold,
        is_unbalance=not args.no_is_unbalance,
    )


if __name__ == "__main__":
    main()
