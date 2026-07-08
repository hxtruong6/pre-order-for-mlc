"""Step 8 of the runtime-scaling benchmark (reviewer 1 response).

Generate FOUR pgfplots/TikZ figures (copy-paste ready for LaTeX):

    {pre-order, partial-order} x {Hamming, Subset}

plus one companion booktabs table (runtime_at_53.tex). Units are SECONDS.

Each figure has:
  * the four main solver curves (GLPK / HiGHS x full / height=2),
  * one dashed power-law TREND line per curve. HiGHS trends are fitted over,
    and drawn across, the full measured K range (the power law holds to K=53).
    GLPK trends are fitted over, and drawn only across, the *tractable* regime
    K<=31 -- NOT extrapolated to K=53, because the measured K=53 points show the
    power law breaks (see below). The solid GLPK line then visibly jumps to its
    real K=53 marker, above where the dashed trend pointed.
  * horizontal baseline lines (BR, CC, CLR, ECC),
  * axis labelled in K (paper notation), log-log, NO interior grid.

Self-contained: all curve data below were measured on the real enron model
(bench_step1 trains the K=53 pairwise RF once; bench_step2* sweeps K by label
subsets; bench_step9/bench_step10 time GLPK directly at K=53) and are frozen
here in milliseconds so the figures regenerate anywhere without the transient
benchmark scratch files.

Measured GLPK per-instance times AT K=53 (from bench_step9/bench_step10):
  * pre-order  height=2: Hamming 622,640 ms, Subset 621,091 ms  (~10 min each)
  * pre-order  full:     DID NOT FINISH within a 48-hour SLURM wall-clock cap
                         (job 409323) -> no K=53 point; curve stops at K=31.
  * partial    height=2: Hamming 361 ms, Subset 351 ms
  * partial    full:     Hamming 525 ms, Subset 521 ms
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

OUTDIR = Path("/home/s2320437/WORK/preorder4MLC/docs/reviewer_response/tikz")
OUTDIR.mkdir(parents=True, exist_ok=True)

ORDER_TITLE = {"preorder": "pre-order", "partial": "partial order"}
METRIC_TITLE = {"Hamming": "Hamming accuracy",
                "Subset": "Subset (0/1 exact-match) accuracy"}

# main curves: (solver, height) -> (tikz color, mark, legend)
CURVES = [
    ("glpk", "full", "red!75!black", "*", "GLPK, full transitivity"),
    ("glpk", "h2", "orange!90!black", "square*", "GLPK, height=2"),
    ("highs", "full", "blue!70!black", "*", "HiGHS, full transitivity"),
    ("highs", "h2", "cyan!70!black", "square*", "HiGHS, height=2"),
]
BASELINES = [
    ("BR", "green!55!black", "Binary Relevance (BR)"),
    ("CC", "teal", "Classifier Chain (CC)"),
    ("CLR", "violet", "Calibrated Label Ranking (CLR)"),
    ("ECC", "brown", "Ensemble of Classifier Chains (ECC)"),
]
SHOW_BASELINE_LABELS = True
BASELINE_LABEL_FONT = "\\fontsize{4}{5}\\selectfont"   # smaller than \tiny

# per-instance baseline inference times (ms), order- and metric-independent
BASELINE_MS = {"BR": 2.548, "CC": 5.189, "CLR": 30.03, "ECC": 137.4}

# GLPK does not scale past its tractable regime; fit + draw its trend only here.
GLPK_FIT_KMAX = 31

# ===== measured per-instance solve times (ms) ==============================
# (solver, height) -> [(K, time_ms), ...]. GLPK K=53 points are the direct
# bench_step9/10 measurements; pre-order GLPK full has none (48h DNF).
DATA = {
    ("preorder", "Hamming"): {
        ("glpk", "full"): [(6, 1.17), (10, 6.96), (14, 35.99), (19, 221.1),
                           (25, 1253.9), (31, 6684.1)],
        ("glpk", "h2"): [(6, 1.11), (10, 18.17), (14, 125.52), (19, 987.29),
                         (25, 5438.2), (31, 21946), (53, 622640)],
        ("highs", "full"): [(6, 4.3652), (10, 9.2426), (14, 26.151), (19, 72.68),
                            (25, 174.45), (31, 364.04), (37, 793.84),
                            (45, 1472.4), (53, 2427.3)],
        ("highs", "h2"): [(6, 1.8111), (10, 4.4508), (14, 9.67), (19, 23.553),
                          (25, 50.792), (31, 100.62), (37, 176.53), (45, 337.27),
                          (53, 579.27)],
    },
    ("preorder", "Subset"): {
        ("glpk", "full"): [(6, 0.87), (10, 7.84), (14, 47.3), (19, 280.6),
                           (25, 1657), (31, 8316.8)],
        ("glpk", "h2"): [(6, 1.13), (10, 17.6), (14, 124.26), (19, 986.67),
                         (25, 5450.6), (31, 21968), (53, 621091)],
        ("highs", "full"): [(6, 2.9996), (10, 8.3643), (14, 23.071), (19, 60.136),
                            (25, 129.27), (31, 245.64), (37, 470.64), (45, 884.73),
                            (53, 1587.2)],
        ("highs", "h2"): [(6, 1.8185), (10, 4.476), (14, 9.6767), (19, 22.593),
                          (25, 50.917), (31, 100.42), (37, 175.77), (45, 334.29),
                          (53, 574.84)],
    },
    ("partial", "Hamming"): {
        ("glpk", "full"): [(6, 0.87594), (10, 1.3623), (14, 3.5519), (19, 9.9031),
                           (25, 25.144), (31, 54.942), (53, 525)],
        ("glpk", "h2"): [(6, 0.36786), (10, 1.0871), (14, 2.6438), (19, 7.3355),
                         (25, 18.24), (31, 38.098), (53, 361)],
        ("highs", "full"): [(6, 3.577), (10, 6.1684), (14, 16.464), (19, 42.865),
                            (25, 89.955), (31, 174.83), (37, 325.61), (45, 599.79),
                            (53, 1102.4)],
        ("highs", "h2"): [(6, 1.6145), (10, 3.1622), (14, 6.396), (19, 14.203),
                          (25, 30.821), (31, 58.542), (37, 102.84), (45, 195.5),
                          (53, 329.86)],
    },
    ("partial", "Subset"): {
        ("glpk", "full"): [(6, 0.4221), (10, 1.3663), (14, 3.5629), (19, 9.896),
                           (25, 25.166), (31, 54.979), (53, 521)],
        ("glpk", "h2"): [(6, 0.37092), (10, 1.104), (14, 2.6534), (19, 7.3475),
                         (25, 18.263), (31, 38.124), (53, 351)],
        ("highs", "full"): [(6, 2.5268), (10, 6.0895), (14, 15.62), (19, 40.948),
                            (25, 81.008), (31, 150.36), (37, 293.86), (45, 531.74),
                            (53, 946.84)],
        ("highs", "h2"): [(6, 1.6253), (10, 3.178), (14, 6.4012), (19, 14.201),
                          (25, 30.811), (31, 58.817), (37, 103.46), (45, 195.92),
                          (53, 331.24)],
    },
}

TABLE_COLS = [("preorder", "Hamming"), ("preorder", "Subset"),
              ("partial", "Hamming"), ("partial", "Subset")]


def curve_s(order, metric, solver, height):
    """Measured points as (K array, seconds array)."""
    pts = sorted(DATA[(order, metric)][(solver, height)])
    Ks = np.array([k for k, _ in pts], dtype=float)
    ys = np.array([v / 1e3 for _, v in pts])   # ms -> s
    return Ks, ys


def fit(Ks, ys):
    slope, intercept = np.polyfit(np.log(Ks), np.log(ys), 1)
    return slope, np.exp(intercept)   # y = coeff * K^slope


def fit_domain(solver, Ks):
    """Fit/draw the dashed trend across the whole range for HiGHS, but only the
    tractable K<=31 regime for GLPK (its power law breaks past there)."""
    if solver == "glpk":
        return 6, min(GLPK_FIT_KMAX, int(Ks.max()))
    return int(Ks.min()), int(Ks.max())


def fmt_s(v):
    """Seconds, ~3 significant figures, table-friendly."""
    if v >= 100:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:.2f}"
    if v >= 0.1:
        return f"{v:.3f}"
    if v >= 0.01:
        return f"{v:.3f}"
    return f"{v:.4f}"


def make_tikz(order, metric):
    lines = []
    P = lines.append
    P("% Auto-generated by scripts/bench_step8_tikz.py -- requires \\usepackage{pgfplots}")
    P("\\begin{tikzpicture}")
    P("\\begin{loglogaxis}[")
    P("    width=10cm, height=7.5cm,")
    P("    xlabel={Number of labels $K$}, ylabel={Avg.\\ time per instance (s)},")
    P(f"    title={{{ORDER_TITLE[order]}, {METRIC_TITLE[metric]}}},")
    P("    xtick={6,10,14,19,25,31,37,45,53},")
    P("    xticklabels={6,10,14,19,25,31,37,45,53},")
    P("    grid=none,")
    P("    xminorticks=false, yminorticks=false,")
    P("    axis x line*=bottom, axis y line*=left,")
    P("    legend pos=north west, legend cell align=left,")
    P("    legend style={font=\\tiny, inner sep=1pt, row sep=-1pt},")
    P("    xmin=5.5, xmax=60,")
    P("]")

    # baselines first (behind the curves); labels parked at the right edge.
    for name, color, _ in BASELINES:
        y = BASELINE_MS[name] / 1e3
        P(f"\\addplot[{color}, dotted, thick, forget plot, domain=6:53, samples=2] {{{y:.4g}}};")
        if SHOW_BASELINE_LABELS:
            P(f"\\node[{color}, font={BASELINE_LABEL_FONT}, anchor=south east] "
              f"at (axis cs:53,{y:.4g}) {{{name} {fmt_s(y)} s}};")

    # main curves: dashed power-law fit (behind) + solid measured markers.
    for solver, height, color, mark, legend in CURVES:
        Ks, ys = curve_s(order, metric, solver, height)
        if len(Ks) == 0:
            continue
        coords = " ".join(f"({int(k)},{v:.5g})" for k, v in zip(Ks, ys))
        d0, d1 = fit_domain(solver, Ks)
        mask = (Ks >= d0) & (Ks <= d1)
        slope, coeff = fit(Ks[mask], ys[mask])
        P(f"\\addplot[{color}, dashed, forget plot, domain={d0}:{d1}, samples=2] "
          f"{{{coeff:.6g}*x^{slope:.4f}}};")
        P(f"\\addplot[{color}, mark={mark}, thick] coordinates {{{coords}}};")
        P(f"\\addlegendentry{{{legend} ($\\propto K^{{{slope:.1f}}}$)}}")

    P("\\end{loglogaxis}")
    P("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def value_at_53_s(order, metric, solver, height):
    """Measured seconds at K=53, or None if that curve has no K=53 point
    (GLPK pre-order full: 48h DNF)."""
    for k, v in DATA[(order, metric)][(solver, height)]:
        if k == 53:
            return v / 1e3
    return None


def make_table():
    body = []
    B = body.append
    B("\\begin{tabular}{lrrrr}")
    B("\\toprule")
    B(" & \\multicolumn{2}{c}{pre-order} & \\multicolumn{2}{c}{partial order}"
      " \\\\")
    B("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    B("Method & Hamming & Subset & Hamming & Subset \\\\")
    B("\\midrule")
    any_dnf = False
    for solver, height, _color, _mark, legend in CURVES:
        cells = []
        for order, metric in TABLE_COLS:
            v = value_at_53_s(order, metric, solver, height)
            if v is None:                 # GLPK could not finish this case
                any_dnf = True
                cells.append("n/a\\textsuperscript{$\\dagger$}")
            else:
                cells.append(fmt_s(v))
        B(f"{legend} & " + " & ".join(cells) + " \\\\")
    B("\\midrule")
    for name, _color, full in BASELINES:      # metric- and order-independent
        v = BASELINE_MS[name] / 1e3
        B(f"{full} & " + " & ".join([fmt_s(v)] * 4) + " \\\\")
    B("\\bottomrule")
    B("\\end{tabular}")
    tabular = "\n".join(body)

    dnf_note = (
        " For the pre-order search GLPK is impractically slow at full "
        "transitivity: a single $K=53$ instance did not finish within a 48-hour "
        "wall-clock limit (vs.\\ under a second for HiGHS), so those two "
        "entries are left \\textsuperscript{$\\dagger$}n/a. The height-2 "
        "restriction is tractable but still about three orders of magnitude "
        "slower than HiGHS. This is precisely why the full-size enron "
        "experiments use HiGHS." if any_dnf else "")
    full = (
        "% Auto-generated by scripts/bench_step8_tikz.py -- needs \\usepackage{booktabs}\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Average per-instance inference time at $K=53$ labels "
        "(enron), in seconds. The four ILP rows are the BOPOs search "
        "times plotted in the runtime figures; the lower block lists the "
        "baseline per-instance inference costs (independent of preference "
        "order and target metric). Measured GLPK entries are timed directly "
        "at $K=53$ (1--10 test instances); HiGHS and baseline "
        f"values are measured over the full test set.{dnf_note}}}\n"
        "\\label{tab:runtime-at-53}\n"
        + tabular + "\n\\end{table}\n")
    return full, tabular


def compile_table_preview(stem, tabular):
    if shutil.which("pdflatex") is None:
        return False
    doc = ("\\documentclass[border=6pt]{standalone}\n"
           "\\usepackage{booktabs}\n\\begin{document}\n"
           + tabular + "\n\\end{document}\n")
    with tempfile.TemporaryDirectory() as td:
        tex = Path(td) / f"{stem}.tex"
        tex.write_text(doc)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        tex.name], cwd=td, capture_output=True, text=True)
        pdf = Path(td) / f"{stem}.pdf"
        if pdf.exists():
            shutil.copy(pdf, OUTDIR / f"{stem}.pdf")
            return True
        return False


def compile_preview(stem, tikz):
    if shutil.which("pdflatex") is None:
        return False
    doc = ("\\documentclass[border=4pt]{standalone}\n"
           "\\usepackage{pgfplots}\\pgfplotsset{compat=1.16}\n"
           "\\begin{document}\n" + tikz + "\\end{document}\n")
    with tempfile.TemporaryDirectory() as td:
        tex = Path(td) / f"{stem}.tex"
        tex.write_text(doc)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-halt-on-error", tex.name], cwd=td,
                           capture_output=True, text=True)
        pdf = Path(td) / f"{stem}.pdf"
        if not pdf.exists():
            print(f"  ! pdflatex failed for {stem}:\n"
                  + "\n".join(r.stdout.splitlines()[-15:]))
            return False
        shutil.copy(pdf, OUTDIR / f"{stem}.pdf")
        if shutil.which("pdftoppm"):
            subprocess.run(["pdftoppm", "-png", "-r", "150",
                           str(OUTDIR / f"{stem}.pdf"),
                           str(OUTDIR / stem)], capture_output=True)
            p1 = OUTDIR / f"{stem}-1.png"
            if p1.exists():
                p1.rename(OUTDIR / f"{stem}.png")
        return True


def main():
    for order in ("preorder", "partial"):
        for metric in ("Hamming", "Subset"):
            stem = f"runtime_{order}_{metric.lower()}"
            tikz = make_tikz(order, metric)
            (OUTDIR / f"{stem}.tex").write_text(tikz)
            ok = compile_preview(stem, tikz)
            print(f"wrote {stem}.tex" + ("  + preview png/pdf" if ok else ""))

    full, tabular = make_table()
    (OUTDIR / "runtime_at_53.tex").write_text(full)
    ok = compile_table_preview("runtime_at_53", tabular)
    print("wrote runtime_at_53.tex" + ("  + preview pdf" if ok else ""))


if __name__ == "__main__":
    main()
