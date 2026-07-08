"""Step 3 of the runtime-scaling benchmark (reviewer 1 response).

Read the scaling CSV + baseline times and render:
  (1) a log-log per-instance ILP solve-time vs L figure (GLPK and HiGHS),
      with baseline per-instance inference times overlaid, and
  (2) a compact markdown runtime table for the rebuttal.

Self-contained: reads the tracked runtime_scaling_data.csv (seconds) and uses
the fixed enron baseline per-instance times, so it regenerates anywhere without
the transient benchmark scratch files. All times are in SECONDS.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA_CSV = REPO / "docs/reviewer_response/runtime_scaling_data.csv"
OUTDIR = REPO / "docs/reviewer_response"
OUTDIR.mkdir(parents=True, exist_ok=True)

# fixed enron (K=53) per-instance baseline inference times, in SECONDS
BASELINE_S = {"BR": 0.002548, "CC": 0.005189, "CLR": 0.03003, "ECC": 0.1374}
N_TEST = 300


def load():
    rows = []
    with DATA_CSV.open() as f:
        for r in csv.DictReader(f):
            r["mean_solve_s"] = float(r["mean_solve_s"])
            r["L"] = int(r["L"])
            rows.append(r)
    meta = {"baseline_per_instance_s": BASELINE_S, "n_test": N_TEST}
    return rows, meta


def series(rows, solver, height, metric, field="mean_solve_s"):
    sel = [r for r in rows if r["solver"] == solver and r["height"] == height
           and r["metric"] == metric]
    sel.sort(key=lambda r: r["L"])
    return [r["L"] for r in sel], [r[field] for r in sel]


def make_figure(rows, meta):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {
        ("glpk", "full"): dict(color="#c0392b", marker="o", ls="-"),
        ("glpk", "h2"): dict(color="#e67e22", marker="s", ls="--"),
        ("highs", "full"): dict(color="#2471a3", marker="o", ls="-"),
        ("highs", "h2"): dict(color="#5dade2", marker="s", ls="--"),
    }
    labels = {
        ("glpk", "full"): "GLPK, full transitivity (IA1/IA3)",
        ("glpk", "h2"): "GLPK, height=2 (IA2/IA4)",
        ("highs", "full"): "HiGHS, full transitivity",
        ("highs", "h2"): "HiGHS, height=2",
    }
    for (solver, height), st in styles.items():
        # average the two target metrics for the headline curve
        Ls = None
        ys = []
        for metric in ("Hamming", "Subset"):
            L, y = series(rows, solver, height, metric)
            if not L:
                continue
            Ls = L
            ys.append(y)
        if Ls is None:
            continue
        ymean = np.mean(ys, axis=0)
        ax.plot(Ls, ymean, label=labels[(solver, height)], lw=1.8,
                markersize=5, **st)

    # baseline per-instance inference (horizontal references)
    bl = meta["baseline_per_instance_s"]
    for name, c in [("BR", "#7f8c8d"), ("CC", "#95a5a6"),
                    ("CLR", "#27ae60"), ("ECC", "#8e44ad")]:
        if name not in bl:
            continue
        ax.axhline(bl[name], color=c, ls=":", lw=1.3)
        ax.text(6.2, bl[name] * 1.05, f"{name} baseline "
                f"({bl[name]:.4g} s/inst)", color=c, fontsize=8, va="bottom")

    # reference power-law slopes anchored at L=14, full-transitivity GLPK
    L14 = 14
    anchor = None
    for r in rows:
        if r["solver"] == "glpk" and r["height"] == "full" \
                and r["metric"] == "Hamming" and r["L"] == L14:
            anchor = r["mean_solve_s"]
    if anchor:
        xs = np.array([6, 53])
        for k, c in [(3, "#bbbbbb"), (4, "#dddddd")]:
            ax.plot(xs, anchor * (xs / L14) ** k, color=c, ls="-", lw=1.0,
                    zorder=0)
            ax.text(53, anchor * (53 / L14) ** k, f" $\\propto L^{k}$",
                    color="#888888", fontsize=8, va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([6, 10, 14, 19, 25, 31, 37, 45, 53])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of labels $L$")
    ax.set_ylabel("Average inference time per test instance (s, log scale)")
    ax.set_title("BOPOs ILP inference cost vs. label-set size\n"
                 "(real enron RF pairwise probabilities, "
                 f"mean over {meta['n_test']} test instances)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"runtime_scaling.{ext}", dpi=150)
    print("wrote", OUTDIR / "runtime_scaling.png")


def make_table(rows, meta):
    lines = []
    lines.append("| L | GLPK full | GLPK h=2 | HiGHS full | HiGHS h=2 |")
    lines.append("|---|-----------|----------|------------|-----------|")
    Ls = sorted({r["L"] for r in rows})
    for L in Ls:
        cells = []
        for solver, height in [("glpk", "full"), ("glpk", "h2"),
                               ("highs", "full"), ("highs", "h2")]:
            vals = [r["mean_solve_s"] for r in rows
                    if r["L"] == L and r["solver"] == solver
                    and r["height"] == height]
            cells.append(f"{np.mean(vals):.4g} s" if vals else "-")
        lines.append(f"| {L} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (OUTDIR / "runtime_table.md").write_text(table + "\n")
    print("\n" + table)
    bl = meta["baseline_per_instance_s"]
    print("\nBaseline per-instance inference (enron K=53):")
    for k, v in bl.items():
        print(f"  {k}: {v:.4g} s")


def main():
    rows, meta = load()
    make_figure(rows, meta)
    make_table(rows, meta)


if __name__ == "__main__":
    main()
