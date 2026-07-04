"""Single-config GLPK pre-order K=53 timing (parallelisable).

Runs ONE (metric, height) pre-order search to completion for a single test
instance and writes the result to its own per-config CSV, so several configs
can grind concurrently on separate cores (GLPK's branch-and-cut is
single-threaded, so more cores per solve does not help; running the four
configs as four processes does). bench_step8_tikz.py globs
"glpk_k53_preorder_p_*.csv" and pools these with the sequential raw log.

Usage:
    python scripts/bench_step10_one.py <Hamming|Subset> <full|h2>
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["GLPK_TIME_LIMIT"] = "100000"      # effectively uncapped

import preorder4mlc.solvers as solvers  # noqa: E402
from preorder4mlc.constants import TargetMetric  # noqa: E402
from preorder4mlc.searching_algorithms import Search_BOPreOs  # noqa: E402

SCRATCH = Path(os.environ.get(
    "PREORDER_BENCH_DIR",
    "/tmp/claude-24679/-home-s2320437-WORK-preorder4MLC/"
    "1acafedf-6413-4348-9250-5e9e5557bcc1/scratchpad"))
K = 53


def main():
    metric_name, hlabel = sys.argv[1], sys.argv[2]
    metric = TargetMetric[metric_name]
    height = None if hlabel == "full" else 2

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

    search = Search_BOPreOs(proba, K, n_test, metric, height=height)
    G, h, A, b, I, B = search._encode_parameters_PRE_ORDER(iv)

    t = time.perf_counter()
    search._solve_one_PRE_ORDER(0, ii, jj, iv, G, h, A, b, I, B)
    dt = time.perf_counter() - t

    out = SCRATCH / f"glpk_k53_preorder_p_{hlabel}_{metric_name}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "metric", "height", "inst", "solve_s"])
        w.writerow(["preorder", metric_name, hlabel, 0, f"{dt:.6f}"])
    print(f"[glpk] preorder {metric_name:7s} {hlabel:4s} inst0 "
          f"{dt:.1f}s ({dt/60:.1f} min) -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
