"""Step 2 of the runtime-scaling benchmark (reviewer 1 response).

Reuse the cached enron pairwise-probability tensor (from step 1) to time
the per-instance BOPOs ILP as the label-set size L grows. For each L we
subsample a fixed subset of L of the 53 real enron labels, slice the real
proba tensor to that subset, and time (a) the one-off constraint encoding
and (b) each per-instance ILP solve. This isolates the L-scaling of the
inference cost using REAL classifier outputs (only L varies).

Solvers: GLPK (paper baseline) and HiGHS (scipy.optimize.milp) are both
timed, so the accuracy/cost trade-off can be reported against either.
"""

import os
import time
from pathlib import Path

import numpy as np

# Bound any single pathological solve so the sweep cannot hang.
os.environ.setdefault("GLPK_TIME_LIMIT", "12")
os.environ.setdefault("HIGHS_TIME_LIMIT", "12")

from preorder4mlc.constants import TargetMetric  # noqa: E402
from preorder4mlc.searching_algorithms import Search_BOPreOs  # noqa: E402

SCRATCH = Path("/tmp/claude-24679/-home-s2320437-WORK-preorder4MLC/"
               "1acafedf-6413-4348-9250-5e9e5557bcc1/scratchpad")

L_GRID = [6, 10, 14, 19, 25, 31, 37, 45, 53]


def timed_for(L):
    # Fewer timed instances at large L where each solve is expensive; the
    # median/p95 are already stable and this bounds total wall time.
    return 30 if L <= 25 else (20 if L <= 37 else 12)
SOLVERS = ["glpk", "highs"]
METRICS = [TargetMetric.Hamming, TargetMetric.Subset]
HEIGHTS = [None, 2]


def subsample(proba_full, L, seed):
    K = proba_full.shape[0]
    if L == K:
        S = np.arange(K)
    else:
        rng = np.random.RandomState(seed)
        S = np.sort(rng.choice(K, L, replace=False))
    n_test = proba_full.shape[2]
    n_cls = proba_full.shape[3]
    return proba_full[np.ix_(S, S, np.arange(n_test), np.arange(n_cls))]


def time_config(proba_sub, L, n_test, metric, height):
    """Return (encode_seconds, per_instance_solve_seconds_list)."""
    search = Search_BOPreOs(proba_sub, L, n_test, metric, height=height)
    # Build indices_vector exactly as PRE_ORDER() does.
    indices_vector, ind = {}, 0
    for i in range(L - 1):
        for j in range(i + 1, L):
            for l in range(4):
                indices_vector[f"{i}_{j}_{l}"] = ind
                ind += 1
    t = time.perf_counter()
    G, h, A, b, I, B = search._encode_parameters_PRE_ORDER(indices_vector)
    encode_t = time.perf_counter() - t

    ii, jj = np.triu_indices(L, k=1)
    solve_times = []
    for n in range(min(timed_for(L), n_test)):
        t = time.perf_counter()
        search._solve_one_PRE_ORDER(n, ii, jj, indices_vector, G, h, A, b, I, B)
        solve_times.append(time.perf_counter() - t)
    return encode_t, solve_times


def main():
    proba_full = np.load(SCRATCH / "enron_proba_preorder.npy")
    n_test = proba_full.shape[2]
    print(f"loaded proba {proba_full.shape}", flush=True)

    rows = []
    for solver in SOLVERS:
        os.environ["PREORDER_SOLVER"] = solver
        for L in L_GRID:
            proba_sub = subsample(proba_full, L, seed=6)
            for metric in METRICS:
                for height in HEIGHTS:
                    enc, st = time_config(proba_sub, L, n_test, metric, height)
                    st = np.array(st)
                    row = {
                        "solver": solver,
                        "L": L,
                        "metric": metric.name,
                        "height": "full" if height is None else "h2",
                        "encode_s": enc,
                        "mean_solve_s": float(st.mean()),
                        "median_solve_s": float(np.median(st)),
                        "p95_solve_s": float(np.percentile(st, 95)),
                        "max_solve_s": float(st.max()),
                        "n_timed": len(st),
                    }
                    rows.append(row)
                    print(
                        f"[{solver}] L={L:2d} {metric.name:7s} "
                        f"{row['height']:4s} enc={enc:6.3f}s "
                        f"mean={row['mean_solve_s']*1e3:8.2f}ms "
                        f"med={row['median_solve_s']*1e3:8.2f}ms "
                        f"p95={row['p95_solve_s']*1e3:9.2f}ms "
                        f"max={row['max_solve_s']*1e3:9.2f}ms",
                        flush=True,
                    )

    import csv
    out = SCRATCH / "scaling_results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
