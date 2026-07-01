"""Step 4 of the runtime-scaling benchmark (reviewer 1 response).

Second figure requested by reviewer 1: instead of only marking curves as
"proportional to L^k", draw explicit anchored reference lines L^3 and L^4,
fit the empirical power-law exponent of each solver/variant curve, and
annotate the fitted exponent directly on the plot. This makes the claim
"GLPK grows ~L^5, HiGHS ~L^3" verifiable at a glance.
"""

import csv
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRATCH = Path("/tmp/claude-24679/-home-s2320437-WORK-preorder4MLC/"
               "1acafedf-6413-4348-9250-5e9e5557bcc1/scratchpad")
OUTDIR = Path("/home/s2320437/WORK/preorder4MLC/results/runtime_scaling")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load():
    rows = []
    with (SCRATCH / "scaling_results.csv").open() as f:
        for r in csv.DictReader(f):
            r["mean_solve_s"] = float(r["mean_solve_s"])
            r["L"] = int(r["L"])
            rows.append(r)
    meta = pickle.load((SCRATCH / "enron_bench_meta.pkl").open("rb"))
    return rows, meta


def curve(rows, solver, height):
    """Mean over the two target metrics per L -> (Ls, times_ms)."""
    Ls = sorted({r["L"] for r in rows
                 if r["solver"] == solver and r["height"] == height})
    ys = []
    for L in Ls:
        vals = [r["mean_solve_s"] for r in rows if r["L"] == L
                and r["solver"] == solver and r["height"] == height]
        ys.append(np.mean(vals) * 1e3)
    return np.array(Ls, dtype=float), np.array(ys)


def fit_exponent(Ls, ys):
    slope, intercept = np.polyfit(np.log(Ls), np.log(ys), 1)
    return slope, intercept


def main():
    rows, meta = load()
    fig, ax = plt.subplots(figsize=(8.5, 6))

    styles = {
        ("glpk", "full"): dict(color="#c0392b", marker="o", ls="-",
                               label="GLPK, full transitivity"),
        ("glpk", "h2"): dict(color="#e67e22", marker="s", ls="-",
                             label="GLPK, height=2"),
        ("highs", "full"): dict(color="#2471a3", marker="o", ls="-",
                                label="HiGHS, full transitivity"),
        ("highs", "h2"): dict(color="#5dade2", marker="s", ls="-",
                              label="HiGHS, height=2"),
    }

    fitted = {}
    for key, st in styles.items():
        Ls, ys = curve(rows, *key)
        if len(Ls) == 0:
            continue
        slope, intercept = fit_exponent(Ls, ys)
        fitted[key] = slope
        lbl = st.pop("label") + f"  (fit $\\propto L^{{{slope:.1f}}}$)"
        ax.plot(Ls, ys, lw=1.9, markersize=5, label=lbl, **st)

    # Explicit anchored reference lines L^3 and L^4.
    # Anchor both at the HiGHS-full point at L=14 so they sit among the data.
    Lh, yh = curve(rows, "highs", "full")
    aL = 14.0
    a_idx = int(np.argmin(np.abs(Lh - aL)))
    anchor_y = yh[a_idx]
    anchor_L = Lh[a_idx]
    xs = np.array([6.0, 53.0])
    for k, c in [(3, "#555555"), (4, "#999999")]:
        ref = anchor_y * (xs / anchor_L) ** k
        ax.plot(xs, ref, color=c, ls="--", lw=1.3, zorder=0)
        ax.text(53, anchor_y * (53 / anchor_L) ** k,
                f"  $L^{k}$ reference", color=c, fontsize=9, va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([6, 10, 14, 19, 25, 31, 37, 45, 53])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of labels $L$ (log scale)")
    ax.set_ylabel("Average inference time per test instance (ms, log scale)")
    ax.set_title("Empirical power-law scaling of the per-instance ILP\n"
                 "(real enron RF pairwise probabilities, "
                 f"mean over {meta['n_test']} test instances)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"runtime_scaling_slopes.{ext}", dpi=150)
    print("wrote", OUTDIR / "runtime_scaling_slopes.png")
    print("\nFitted exponents (time ~ L^x):")
    for key, s in fitted.items():
        print(f"  {key[0]:5s} {key[1]:4s}: L^{s:.2f}")


if __name__ == "__main__":
    main()
