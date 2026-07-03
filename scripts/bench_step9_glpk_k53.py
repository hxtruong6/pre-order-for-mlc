"""Step 9 of the runtime-scaling benchmark (reviewer 1 response).

Measure GLPK at the real enron size K=53 directly, on a small sample of
instances, so the K=53 table no longer relies on extrapolation. GLPK is very
slow at K=53 (pre-order height=2 is minutes per instance), so we time only
N_TIMED instances and write each solve time immediately, fastest
configurations first: partial order, then pre-order full transitivity, then
pre-order height=2 (the slowest). Rerun scripts/bench_step8_tikz.py afterwards
to fold the measured means into runtime_at_53.tex.
"""

import csv
import os
import time
from pathlib import Path

import numpy as np

# no solve-time cap: we want the true time-to-optimality, not a truncated one
os.environ["GLPK_TIME_LIMIT"] = "100000"

import preorder4mlc.solvers as solvers  # noqa: E402
from preorder4mlc.constants import TargetMetric  # noqa: E402
from preorder4mlc.searching_algorithms import (  # noqa: E402
    Search_BOParOs, Search_BOPreOs)

SCRATCH = Path("/tmp/claude-24679/-home-s2320437-WORK-preorder4MLC/"
               "1acafedf-6413-4348-9250-5e9e5557bcc1/scratchpad")
K = 53
N_TIMED = 10
METRICS = [TargetMetric.Hamming, TargetMetric.Subset]
HEIGHTS = [None, 2]

# fastest configurations first so most cells fill quickly; pre-order h2 last
PLAN = [
    ("partial", SCRATCH / "enron_proba_partial.npy", 3, None),
    ("partial", SCRATCH / "enron_proba_partial.npy", 3, 2),
    ("preorder", SCRATCH / "enron_proba_preorder.npy", 4, None),
    ("preorder", SCRATCH / "enron_proba_preorder.npy", 4, 2),
]


def indices_vector(n_rel):
    iv, ind = {}, 0
    for i in range(K - 1):
        for j in range(i + 1, K):
            for l in range(n_rel):
                iv[f"{i}_{j}_{l}"] = ind
                ind += 1
    return iv


def main():
    solvers.set_solver("glpk")
    out = SCRATCH / "glpk_k53_raw.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "metric", "height", "inst", "solve_s"])
        for order, path, n_rel, height in PLAN:
            proba = np.load(path)
            n_test = proba.shape[2]
            iv = indices_vector(n_rel)
            ii, jj = np.triu_indices(K, k=1)
            hlabel = "full" if height is None else "h2"
            for metric in METRICS:
                if order == "preorder":
                    search = Search_BOPreOs(proba, K, n_test, metric, height=height)
                    G, h, A, b, I, B = search._encode_parameters_PRE_ORDER(iv)
                    solve_one = lambda n: search._solve_one_PRE_ORDER(
                        n, ii, jj, iv, G, h, A, b, I, B)
                else:
                    search = Search_BOParOs(proba, K, n_test, metric, height=height)
                    G, h, A, b, I, B = search._encode_parameters_PARTIAL_ORDER(iv)
                    solve_one = lambda n: search._solve_one_PARTIAL_ORDER(
                        n, ii, jj, iv, G, h, A, b, I, B)
                for n in range(min(N_TIMED, n_test)):
                    t = time.perf_counter()
                    solve_one(n)
                    dt = time.perf_counter() - t
                    w.writerow([order, metric.name, hlabel, n, f"{dt:.6f}"])
                    f.flush()
                    print(f"[glpk] {order:8s} {metric.name:7s} {hlabel:4s} "
                          f"inst{n:2d} {dt*1e3:10.1f}ms", flush=True)
    print("DONE -> glpk_k53_raw.csv", flush=True)


if __name__ == "__main__":
    main()
