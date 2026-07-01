"""Step 2c: focused L=53 (real enron K) HiGHS pass.

The full-transitivity ILP at L=53 is the single hardest case (140k
constraints); we time a smaller sample of instances with a strict cap so
the headline enron number completes promptly, then merge GLPK(L<=31) +
HiGHS(L<=45) + HiGHS(L=53) into scaling_results.csv.
"""

import csv
import os
import time
from pathlib import Path

import numpy as np

os.environ["PREORDER_SOLVER"] = "highs"
os.environ["HIGHS_TIME_LIMIT"] = "15"

from preorder4mlc.constants import TargetMetric  # noqa: E402
from preorder4mlc.searching_algorithms import Search_BOPreOs  # noqa: E402

SCRATCH = Path("/tmp/claude-24679/-home-s2320437-WORK-preorder4MLC/"
               "1acafedf-6413-4348-9250-5e9e5557bcc1/scratchpad")
FIELDS = ["solver", "L", "metric", "height", "encode_s", "mean_solve_s",
          "median_solve_s", "p95_solve_s", "max_solve_s", "n_timed"]
TIMED = 8


def main():
    proba = np.load(SCRATCH / "enron_proba_preorder.npy")  # (53,53,60,4)
    L = 53
    n_test = proba.shape[2]
    rows = []
    for metric in [TargetMetric.Hamming, TargetMetric.Subset]:
        for height in [None, 2]:
            search = Search_BOPreOs(proba, L, n_test, metric, height=height)
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
            for n in range(min(TIMED, n_test)):
                t = time.perf_counter()
                search._solve_one_PRE_ORDER(n, ii, jj, iv, G, h, A, b, I, B)
                st.append(time.perf_counter() - t)
                print(f"  L=53 {metric.name} h={height} inst{n} "
                      f"{st[-1]:.2f}s", flush=True)
            st = np.array(st)
            rows.append(dict(solver="highs", L=L, metric=metric.name,
                             height="full" if height is None else "h2",
                             encode_s=enc, mean_solve_s=float(st.mean()),
                             median_solve_s=float(np.median(st)),
                             p95_solve_s=float(np.percentile(st, 95)),
                             max_solve_s=float(st.max()), n_timed=len(st)))
            print(f"[highs] L=53 {metric.name:7s} "
                  f"{'full' if height is None else 'h2':4s} "
                  f"mean={st.mean()*1e3:.1f}ms", flush=True)

    with (SCRATCH / "scaling_highs_L53.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # merge everything
    merged = []
    for fn in ("scaling_glpk_partial.csv", "scaling_highs.csv",
               "scaling_highs_L53.csv"):
        with (SCRATCH / fn).open() as f:
            merged.extend(list(csv.DictReader(f)))
    with (SCRATCH / "scaling_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(merged)
    print(f"merged {len(merged)} rows -> scaling_results.csv", flush=True)


if __name__ == "__main__":
    main()
