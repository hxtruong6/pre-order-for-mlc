"""Step 2b: HiGHS full-grid pass for the runtime-scaling benchmark.

GLPK (paper baseline solver) becomes impractical for L>=37 (a single
height-2 solve already exceeds 20 s at L=31), so its curve is capped at
L<=31 and recovered separately. HiGHS (scipy.optimize.milp) respects its
time limit and is the recommended solver; we time it across the full
L grid up to the real enron K=53. Rows are appended incrementally and
merged with the recovered GLPK rows into scaling_results.csv.
"""

import csv
import os
import time
from pathlib import Path

import numpy as np

os.environ["PREORDER_SOLVER"] = "highs"
os.environ.setdefault("HIGHS_TIME_LIMIT", "12")

from preorder4mlc.constants import TargetMetric  # noqa: E402
from preorder4mlc.searching_algorithms import Search_BOPreOs  # noqa: E402

SCRATCH = Path("/tmp/claude-24679/-home-s2320437-WORK-preorder4MLC/"
               "1acafedf-6413-4348-9250-5e9e5557bcc1/scratchpad")

L_GRID = [6, 10, 14, 19, 25, 31, 37, 45, 53]
METRICS = [TargetMetric.Hamming, TargetMetric.Subset]
HEIGHTS = [None, 2]
FIELDS = ["solver", "L", "metric", "height", "encode_s", "mean_solve_s",
          "median_solve_s", "p95_solve_s", "max_solve_s", "n_timed"]


def timed_for(L):
    return 30 if L <= 25 else (20 if L <= 37 else 12)


def subsample(proba_full, L, seed):
    K = proba_full.shape[0]
    S = np.arange(K) if L == K else np.sort(
        np.random.RandomState(seed).choice(K, L, replace=False))
    return proba_full[np.ix_(S, S, np.arange(proba_full.shape[2]),
                             np.arange(proba_full.shape[3]))]


def time_config(proba_sub, L, n_test, metric, height):
    search = Search_BOPreOs(proba_sub, L, n_test, metric, height=height)
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
    st = []
    for n in range(min(timed_for(L), n_test)):
        t = time.perf_counter()
        search._solve_one_PRE_ORDER(n, ii, jj, indices_vector, G, h, A, b, I, B)
        st.append(time.perf_counter() - t)
    return encode_t, np.array(st)


def main():
    proba_full = np.load(SCRATCH / "enron_proba_preorder.npy")
    n_test = proba_full.shape[2]
    out = SCRATCH / "scaling_highs.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for L in L_GRID:
            proba_sub = subsample(proba_full, L, seed=6)
            for metric in METRICS:
                for height in HEIGHTS:
                    enc, st = time_config(proba_sub, L, n_test, metric, height)
                    row = dict(solver="highs", L=L, metric=metric.name,
                               height="full" if height is None else "h2",
                               encode_s=enc, mean_solve_s=float(st.mean()),
                               median_solve_s=float(np.median(st)),
                               p95_solve_s=float(np.percentile(st, 95)),
                               max_solve_s=float(st.max()), n_timed=len(st))
                    w.writerow(row)
                    f.flush()
                    print(f"[highs] L={L:2d} {metric.name:7s} {row['height']:4s} "
                          f"enc={enc:6.3f}s mean={row['mean_solve_s']*1e3:8.2f}ms "
                          f"med={row['median_solve_s']*1e3:8.2f}ms "
                          f"p95={row['p95_solve_s']*1e3:9.2f}ms "
                          f"max={row['max_solve_s']*1e3:9.2f}ms", flush=True)

    # Merge recovered GLPK rows (L<=31) + HiGHS rows -> scaling_results.csv
    merged = []
    for fn in ("scaling_glpk_partial.csv", "scaling_highs.csv"):
        with (SCRATCH / fn).open() as f:
            merged.extend(list(csv.DictReader(f)))
    with (SCRATCH / "scaling_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(merged)
    print(f"merged {len(merged)} rows -> scaling_results.csv", flush=True)


if __name__ == "__main__":
    main()
