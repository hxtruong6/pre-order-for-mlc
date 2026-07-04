"""Step 10 of the runtime-scaling benchmark (reviewer 1 response).

Measure GLPK on the PRE-ORDER search at the real enron size K=53 by running a
single instance per configuration to completion (no time cap). These solves
are extremely slow (tens of minutes each; height=2 far worse), so we time just
N_TIMED=1 instance and write each result the moment it finishes, fastest first
(full transitivity before height=2). Rerun scripts/bench_step8_tikz.py
afterwards to fold the measured value(s) into runtime_at_53.tex.
"""

import csv
import os
import time
from pathlib import Path

import numpy as np

os.environ["GLPK_TIME_LIMIT"] = "100000"      # effectively uncapped

import preorder4mlc.solvers as solvers  # noqa: E402
from preorder4mlc.constants import TargetMetric  # noqa: E402
from preorder4mlc.searching_algorithms import Search_BOPreOs  # noqa: E402

SCRATCH = Path("/tmp/claude-24679/-home-s2320437-WORK-preorder4MLC/"
               "1acafedf-6413-4348-9250-5e9e5557bcc1/scratchpad")
K = 53
N_TIMED = 1
METRICS = [TargetMetric.Hamming, TargetMetric.Subset]
HEIGHTS = [None, 2]                            # full first (faster), then h2


def main():
    solvers.set_solver("glpk")
    proba = np.load(SCRATCH / "enron_proba_preorder.npy")
    n_test = proba.shape[2]
    iv, ind = {}, 0
    for i in range(K - 1):
        for j in range(i + 1, K):
            for l in range(4):
                iv[f"{i}_{j}_{l}"] = ind
                ind += 1
    ii, jj = np.triu_indices(K, k=1)

    out = SCRATCH / "glpk_k53_preorder_raw.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "metric", "height", "inst", "solve_s"])
        for height in HEIGHTS:
            hlabel = "full" if height is None else "h2"
            for metric in METRICS:
                search = Search_BOPreOs(proba, K, n_test, metric, height=height)
                G, h, A, b, I, B = search._encode_parameters_PRE_ORDER(iv)
                for n in range(N_TIMED):
                    t = time.perf_counter()
                    search._solve_one_PRE_ORDER(n, ii, jj, iv, G, h, A, b, I, B)
                    dt = time.perf_counter() - t
                    w.writerow(["preorder", metric.name, hlabel, n, f"{dt:.6f}"])
                    f.flush()
                    print(f"[glpk] preorder {metric.name:7s} {hlabel:4s} "
                          f"inst{n} {dt:.1f}s ({dt/60:.1f} min)", flush=True)
    print("DONE -> glpk_k53_preorder_raw.csv", flush=True)


if __name__ == "__main__":
    main()
