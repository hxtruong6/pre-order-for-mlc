"""Step 6 of the runtime-scaling benchmark (reviewer 1 response).

Train the REAL enron (K=53) pairwise RF model for the PARTIAL_ORDER variant
(three classes per pair) and dump the (53,53,N_TEST,3) probabilistic-prediction
tensor. Mirrors bench_step1 (which did PRE_ORDER, four classes per pair) using
the identical split so the two variants are directly comparable.
"""

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

N_TEST = 300


def main():
    t0 = time.perf_counter()
    loader = Datasets4Experiments(
        data_path=str(ROOT / "data") + "/",
        data_files=[{"dataset_name": "enron.arff", "n_labels_set": 53}],
    )
    loader.load_datasets()
    X, Y, _ = loader.get_datasets()[0]
    n_labels = Y.shape[1]

    rng = np.random.RandomState(0)          # same split as bench_step1
    perm = rng.permutation(len(X))
    cut = int(len(X) * 0.7)
    train_idx, test_idx = perm[:cut], perm[cut:][:N_TEST]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test = X[test_idx]
    print(f"enron K={n_labels} train={X_train.shape} test={X_test.shape}",
          flush=True)

    bl = BaseLearnerName.RF.value
    t = time.perf_counter()
    m = PredictBOPOs(bl, preference_order=PreferenceOrder.PARTIAL_ORDER)
    m.fit(X_train, Y_train)
    print(f"[fit pairwise PARTIAL_ORDER] {time.perf_counter()-t:.1f}s", flush=True)

    t = time.perf_counter()
    proba = m.predict_proba(X_test, n_labels)  # (53,53,N_TEST,3)
    print(f"[predict_proba] {time.perf_counter()-t:.1f}s tensor={proba.shape}",
          flush=True)

    np.save(OUT / "enron_proba_partial.npy", proba)
    print(f"DONE in {time.perf_counter()-t0:.1f}s -> enron_proba_partial.npy",
          flush=True)


if __name__ == "__main__":
    main()
