"""Combined runtime-scaling sweep (GLPK L<=31 + HiGHS full grid).

Times the average per-instance BOPOs ILP solve on real enron pairwise
probabilities as the label-set size L grows, for both solvers and both
heights. GLPK is capped at L<=31 because a single instance already exceeds
practical time beyond that. Writes rows incrementally, then merges into
scaling_results.csv.
"""

import csv
import os
import time
from pathlib import Path

import numpy as np

os.environ["GLPK_TIME_LIMIT"] = "20"
os.environ["HIGHS_TIME_LIMIT"] = "20"

from preorder4mlc.constants import TargetMetric  # noqa: E402
import preorder4mlc.solvers as solvers  # noqa: E402
from preorder4mlc.searching_algorithms import Search_BOPreOs  # noqa: E402

SCRATCH = Path("/tmp/claude-24679/-home-s2320437-WORK-preorder4MLC/"
               "1acafedf-6413-4348-9250-5e9e5557bcc1/scratchpad")
FIELDS = ["solver", "L", "metric", "height", "encode_s", "mean_solve_s",
          "median_solve_s", "p95_solve_s", "max_solve_s", "n_timed"]
METRICS = [TargetMetric.Hamming, TargetMetric.Subset]
HEIGHTS = [None, 2]

# (solver, L_grid, timed_fn)
PLAN = [
    ("highs", [6, 10, 14, 19, 25, 31, 37, 45, 53],
     lambda L: 100 if L <= 31 else (60 if L <= 45 else 40)),
    ("glpk", [6, 10, 14, 19, 25, 31],
     lambda L: 50 if L <= 19 else (30 if L <= 25 else 20)),
]


def subsample(proba_full, L, seed=6):
    K = proba_full.shape[0]
    S = np.arange(K) if L == K else np.sort(
        np.random.RandomState(seed).choice(K, L, replace=False))
    return proba_full[np.ix_(S, S, np.arange(proba_full.shape[2]),
                             np.arange(proba_full.shape[3]))]


def time_config(proba_sub, L, n_test, metric, height, n_timed):
    search = Search_BOPreOs(proba_sub, L, n_test, metric, height=height)
    iv, ind = {}, 0
    for i in range(L - 1):
        for j in range(i + 1, L):
            for l in range(4):
                iv[f"{i}_{j}_{l}"] = ind
                ind += 1
    t = time.perf_counter()
    G, h, A, b, I, B = search._encode_parameters_PRE_ORDER(iv)
    enc = time.perf_counter() - t
    ii, jj = np.triu_indices(L, k=1)
    st = []
    for n in range(min(n_timed, n_test)):
        t = time.perf_counter()
        search._solve_one_PRE_ORDER(n, ii, jj, iv, G, h, A, b, I, B)
        st.append(time.perf_counter() - t)
    return enc, np.array(st)


def main():
    proba_full = np.load(SCRATCH / "enron_proba_preorder.npy")
    n_test = proba_full.shape[2]
    print(f"loaded proba {proba_full.shape}", flush=True)

    out = SCRATCH / "scaling_results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for solver, grid, timed_fn in PLAN:
            solvers.set_solver(solver)
            for L in grid:
                proba_sub = subsample(proba_full, L)
                for metric in METRICS:
                    for height in HEIGHTS:
                        enc, st = time_config(proba_sub, L, n_test, metric,
                                              height, timed_fn(L))
                        row = dict(solver=solver, L=L, metric=metric.name,
                                   height="full" if height is None else "h2",
                                   encode_s=enc, mean_solve_s=float(st.mean()),
                                   median_solve_s=float(np.median(st)),
                                   p95_solve_s=float(np.percentile(st, 95)),
                                   max_solve_s=float(st.max()), n_timed=len(st))
                        w.writerow(row)
                        f.flush()
                        print(f"[{solver}] L={L:2d} {metric.name:7s} "
                              f"{row['height']:4s} n={len(st):3d} "
                              f"mean={row['mean_solve_s']*1e3:8.2f}ms "
                              f"med={row['median_solve_s']*1e3:8.2f}ms",
                              flush=True)
    print("DONE -> scaling_results.csv", flush=True)


if __name__ == "__main__":
    main()
