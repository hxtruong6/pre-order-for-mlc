"""Step 5 of the runtime-scaling benchmark (reviewer 1 response).

Reviewer 1 asked to separate the two target metrics into their own charts.
This produces TWO figures, one per target metric:

  * runtime_scaling_hamming.png  -- Hamming accuracy
  * runtime_scaling_subset.png   -- Subset (0/1 exact-match) accuracy

Each figure shows, for that single target metric:
  * the four ILP curves (GLPK / HiGHS x full / height=2) with fitted L^x,
  * explicit anchored L^3 and L^4 reference lines,
  * horizontal baseline lines (BR, CC, CLR, ECC) for context.
"""

import csv
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

SCRATCH = Path("/tmp/claude-24679/-home-s2320437-WORK-preorder4MLC/"
               "1acafedf-6413-4348-9250-5e9e5557bcc1/scratchpad")
OUTDIR = Path("/home/s2320437/WORK/preorder4MLC/results/runtime_scaling")
OUTDIR.mkdir(parents=True, exist_ok=True)

# metric key in CSV -> (filename stem, human title)
METRICS = {
    "Hamming": ("runtime_scaling_hamming", "Hamming accuracy"),
    "Subset": ("runtime_scaling_subset", "Subset (0/1 exact-match) accuracy"),
}

BASELINE_STYLE = {
    "BR": ("#27ae60", "Binary Relevance (BR)"),
    "CC": ("#16a085", "Classifier Chain (CC)"),
    "CLR": ("#8e44ad", "Calibrated Label Ranking (CLR)"),
    "ECC": ("#d35400", "Ensemble of Classifier Chains (ECC)"),
}

ILP_STYLE = {
    ("glpk", "full"): dict(color="#c0392b", marker="o", ls="-",
                           label="GLPK, full transitivity"),
    ("glpk", "h2"): dict(color="#e67e22", marker="s", ls="-",
                         label="GLPK, height=2"),
    ("highs", "full"): dict(color="#2471a3", marker="o", ls="-",
                            label="HiGHS, full transitivity"),
    ("highs", "h2"): dict(color="#5dade2", marker="s", ls="-",
                          label="HiGHS, height=2"),
}


def load():
    rows = []
    with (SCRATCH / "scaling_results.csv").open() as f:
        for r in csv.DictReader(f):
            r["mean_solve_s"] = float(r["mean_solve_s"])
            r["L"] = int(r["L"])
            rows.append(r)
    meta = pickle.load((SCRATCH / "enron_bench_meta.pkl").open("rb"))
    return rows, meta


def curve(rows, solver, height, metric):
    """Single-metric curve -> (Ls, times_ms)."""
    sel = [r for r in rows if r["solver"] == solver
           and r["height"] == height and r["metric"] == metric]
    Ls = sorted({r["L"] for r in sel})
    ys = [next(r["mean_solve_s"] for r in sel if r["L"] == L) * 1e3
          for L in Ls]
    return np.array(Ls, dtype=float), np.array(ys)


def make_figure(rows, meta, metric_key, stem, title):
    fig, ax = plt.subplots(figsize=(8.5, 6))

    # ILP curves for this metric.
    for key, st in ILP_STYLE.items():
        Ls, ys = curve(rows, key[0], key[1], metric_key)
        if len(Ls) == 0:
            continue
        slope, _ = np.polyfit(np.log(Ls), np.log(ys), 1)
        st = dict(st)
        lbl = st.pop("label") + f"  (fit $\\propto L^{{{slope:.1f}}}$)"
        ax.plot(Ls, ys, lw=1.9, markersize=5, label=lbl, **st)

    # Anchored L^3 and L^4 reference lines (anchor on HiGHS-full at L=14).
    Lh, yh = curve(rows, "highs", "full", metric_key)
    a = int(np.argmin(np.abs(Lh - 14.0)))
    anchor_y, anchor_L = yh[a], Lh[a]
    xs = np.array([6.0, 53.0])
    for k, c in [(3, "#555555"), (4, "#999999")]:
        ax.plot(xs, anchor_y * (xs / anchor_L) ** k,
                color=c, ls="--", lw=1.3, zorder=0)
        ax.text(53, anchor_y * (53 / anchor_L) ** k,
                f"  $L^{k}$ reference", color=c, fontsize=9, va="center")

    # Baseline horizontal lines (per-instance, ms). Metric-independent.
    for name, sec in meta["baseline_per_instance_s"].items():
        c, lbl = BASELINE_STYLE[name]
        ax.axhline(sec * 1e3, color=c, ls=":", lw=1.4)
        ax.text(6, sec * 1e3, f" {lbl}: {sec*1e3:.1f} ms",
                color=c, fontsize=8, va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([6, 10, 14, 19, 25, 31, 37, 45, 53])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Number of labels $L$ (log scale)")
    ax.set_ylabel("Average time per test instance (ms, log scale)")
    ax.set_title(f"Per-instance ILP runtime scaling -- {title}\n"
                 "(real enron RF pairwise probabilities, "
                 f"mean over {meta['n_test']} test instances)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", dpi=150)
    plt.close(fig)
    print("wrote", OUTDIR / f"{stem}.png")


def main():
    rows, meta = load()
    for metric_key, (stem, title) in METRICS.items():
        make_figure(rows, meta, metric_key, stem, title)
        print(f"  fitted exponents ({metric_key}):")
        for key in ILP_STYLE:
            Ls, ys = curve(rows, key[0], key[1], metric_key)
            if len(Ls):
                s, _ = np.polyfit(np.log(Ls), np.log(ys), 1)
                print(f"    {key[0]:5s} {key[1]:4s}: L^{s:.2f}")


if __name__ == "__main__":
    main()
