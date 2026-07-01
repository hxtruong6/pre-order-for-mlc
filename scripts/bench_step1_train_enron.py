"""Step 1 of the runtime-scaling benchmark (reviewer 1 response).

Train the REAL enron (K=53) pairwise RF model once, dump the pairwise
probabilistic-prediction tensor and the baseline (BR/CC/CLR) per-instance
inference times to a cache under scratch. Step 2 reuses the cached tensor
to time the BOPOs ILP as a function of the label-set size L (by subsampling
label subsets), so the model is trained only once.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

from preorder4mlc.constants import BaseLearnerName
from preorder4mlc.datasets4experiments import Datasets4Experiments
from preorder4mlc.inference_models import PredictBOPOs, PreferenceOrder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
OUT = Path("/tmp/claude-24679/-home-s2320437-WORK-preorder4MLC/"
           "1acafedf-6413-4348-9250-5e9e5557bcc1/scratchpad")
OUT.mkdir(parents=True, exist_ok=True)

N_TEST = 300  # test instances whose proba we keep (for a robust per-instance mean)


def main():
    t0 = time.perf_counter()
    loader = Datasets4Experiments(
        data_path=str(ROOT / "data") + "/",
        data_files=[{"dataset_name": "enron.arff", "n_labels_set": 53}],
    )
    loader.load_datasets()
    X, Y, _ = loader.get_datasets()[0]
    n_labels = Y.shape[1]
    print(f"enron loaded: X={X.shape} Y={Y.shape} K={n_labels}", flush=True)

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(X))
    cut = int(len(X) * 0.7)
    train_idx, test_idx = perm[:cut], perm[cut:][:N_TEST]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test = X[test_idx]
    print(f"train={X_train.shape} test={X_test.shape}", flush=True)

    bl = BaseLearnerName.RF.value

    # --- BOPOs pairwise model: fit + predict_proba (PRE_ORDER, 4 classes) ---
    t = time.perf_counter()
    m = PredictBOPOs(bl, preference_order=PreferenceOrder.PRE_ORDER)
    m.fit(X_train, Y_train)
    fit_t = time.perf_counter() - t
    print(f"[fit pairwise PRE_ORDER] {fit_t:.1f}s", flush=True)

    t = time.perf_counter()
    proba = m.predict_proba(X_test, n_labels)  # (53,53,N_TEST,4)
    pp_t = time.perf_counter() - t
    print(f"[predict_proba] {pp_t:.1f}s tensor={proba.shape}", flush=True)

    # --- Baselines: fit + timed predict over the same test set ---
    baseline_times = {}
    for name, fit_fn, pred_fn in [
        ("BR", "fit_BR", "predict_BR"),
        ("CC", "fit_CC", "predict_CC"),
        ("CLR", "fit_CLR", "predict_CLR"),
    ]:
        mb = PredictBOPOs(bl, preference_order=PreferenceOrder.PRE_ORDER)
        getattr(mb, fit_fn)(X_train, Y_train)
        t = time.perf_counter()
        getattr(mb, pred_fn)(X_test, n_labels)
        dt = time.perf_counter() - t
        baseline_times[name] = dt / len(X_test)  # per-instance seconds
        print(f"[baseline {name}] total={dt:.3f}s per_inst={dt/len(X_test)*1e3:.3f}ms",
              flush=True)

    # --- ECC baseline (Ensemble of 10 Classifier Chains): fit chains once,
    #     time prediction only (to match the predict-only timing above). ---
    from train_ecc import ClassifierChain, _make_base_learner, _to_dense_int
    from preorder4mlc.constants import RANDOM_STATE
    rng = np.random.RandomState(RANDOM_STATE)
    chains = []
    for k in range(10):
        perm = rng.permutation(n_labels)
        inv = np.argsort(perm)
        idx = rng.randint(0, len(X_train), size=len(X_train))
        ch = ClassifierChain(
            classifier=_make_base_learner("rf", random_state=RANDOM_STATE + k,
                                          is_unbalance=True),
            require_dense=[True, True])
        ch.fit(X_train[idx], Y_train[idx][:, perm])
        chains.append((ch, inv))
    t = time.perf_counter()
    for ch, inv in chains:
        _to_dense_int(ch.predict(X_test))
        try:
            ch.predict_proba(X_test)  # matches train_ecc: guarded, may fail
        except Exception:
            pass
    dt_ecc = time.perf_counter() - t
    baseline_times["ECC"] = dt_ecc / len(X_test)
    print(f"[baseline ECC] total={dt_ecc:.3f}s per_inst={dt_ecc/len(X_test)*1e3:.3f}ms",
          flush=True)

    np.save(OUT / "enron_proba_preorder.npy", proba)
    with (OUT / "enron_bench_meta.pkl").open("wb") as f:
        pickle.dump({
            "n_labels": n_labels,
            "n_test": len(X_test),
            "fit_time": fit_t,
            "predict_proba_time": pp_t,
            "baseline_per_instance_s": baseline_times,
        }, f)
    print(f"DONE in {time.perf_counter()-t0:.1f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
