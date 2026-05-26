#!/usr/bin/env python3
"""Emit ``paper_revision/results_revision.tex`` from the figures tree.

Mirrors the layout produced by ``scripts/paper_revision/build_panels.py``:

    paper_revision/figures/<learner>/<aggregate|per_dataset/<ds>>/<bv|pa>/<metric>.pdf

Each (learner, scope, prediction_type) tuple becomes one ``\\begin{table}``
laid out as a 3-column ``tabular`` of ``\\includegraphics`` panels. The .tex
file is generated, so re-running this script after re-running ``build_panels.py``
is the entire maintenance loop.

Usage::

    python scripts/paper_revision/build_tex.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURES = REPO_ROOT / "paper_revision" / "figures"
DEFAULT_TEX = REPO_ROOT / "paper_revision" / "results_revision.tex"
DEFAULT_STATS = REPO_ROOT / "paper_revision" / "dataset_stats.json"

DATASETS = [
    # 6 original-paper datasets.
    "GpositivePseAAC",
    "emotions",
    "scene",
    "PlantPseAAC",
    "HumanPseAAC",
    "Yeast",
    # 3 revision-extension datasets.
    "birds",
    "medical",
    "enron",
]
DATASET_DISPLAY = {
    "GpositivePseAAC": "GpositivePseAAC ($K=4$)",
    "emotions": "Emotions ($K=6$)",
    "scene": "Scene ($K=6$)",
    "PlantPseAAC": "PlantPseAAC ($K=12$)",
    "HumanPseAAC": "HumanPseAAC ($K=14$)",
    "Yeast": "Yeast ($K=14$)",
    "birds": "birds ($K=19$)",
    "medical": "medical ($K=45$)",
    "enron": "enron ($K=53$)",
}
PTYPE_LABEL = {
    "bv": "BinaryVector",
    "pa": "PartialAbstention",
    "sv": "ScoreVector",
}
PTYPE_TITLE = {
    "bv": "Standard MLC predictions",
    "pa": "Partial abstention",
    "sv": "Score-vector ranking metrics (AUROC, AUPRC, etc.)",
}
PTYPE_KEYS = ("bv", "pa", "sv")

COLS_PER_ROW = 3
PANEL_WIDTH = r"0.31\linewidth"


_PREAMBLE_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}

\graphicspath{{__GRAPHICS_PATH__/}}
\newcommand{\mpanel}[1]{\includegraphics[width=__PANEL_WIDTH__]{#1}}

\title{__TITLE__}
\author{}
\date{}
"""


def _make_preamble(figures_dir_name: str, style: str) -> str:
    suffix = " --- enhanced style" if style == "enhanced" else " --- original style"
    return (
        _PREAMBLE_TEMPLATE
        .replace("__GRAPHICS_PATH__", figures_dir_name)
        .replace("__PANEL_WIDTH__", PANEL_WIDTH)
        .replace("__TITLE__", "Results --- revised submission" + suffix)
    )


_METHOD_BLOCK_A_B = r"""\section{Method encoding}\label{sec:method_encoding}
For every panel that follows, the x-axis encodes the classifier as one of
12 methods, grouped into three blocks:

\paragraph{Block A --- Order-based predictors (ours, indices 1--8).}
Each name has the form \texttt{<order>-<target>-<height>}:
\emph{order} $\in$ \{PA = partial order, PR = pre-order\};
\emph{target} $\in$ \{H = Hamming, S = Subset~0/1\};
\emph{height} $\in$ \{2, $\infty$ (omitted in the label)\}.
\begin{itemize}
  \item 1\,=\,\textbf{PA-H-2}: partial order, Hamming target, height $\le 2$.
  \item 2\,=\,\textbf{PA-H}: partial order, Hamming target, unrestricted height.
  \item 3\,=\,\textbf{PA-S-2}: partial order, Subset 0/1 target, height $\le 2$.
  \item 4\,=\,\textbf{PA-S}: partial order, Subset 0/1, unrestricted.
  \item 5\,=\,\textbf{PR-H-2}: pre-order, Hamming, height $\le 2$.
  \item 6\,=\,\textbf{PR-H}: pre-order, Hamming, unrestricted.
  \item 7\,=\,\textbf{PR-S-2}: pre-order, Subset 0/1, height $\le 2$.
  \item 8\,=\,\textbf{PR-S}: pre-order, Subset 0/1, unrestricted.
\end{itemize}

\paragraph{Block B --- Standard MLC baselines from the original paper
(indices 9--11).}
\begin{itemize}
  \item 9\,=\,\textbf{BR} (Binary Relevance): $K$ independent binary
        classifiers, one per label.
  \item 10\,=\,\textbf{CC} (Classifier Chain): sequential chain that uses
        previous labels as features.
  \item 11\,=\,\textbf{CLR} (Calibrated Label Ranking): pairwise label
        ranking with a calibration label that separates relevant from
        irrelevant.
\end{itemize}

\paragraph{Block C --- Extended baseline added for the revision
(index 12).}
\begin{itemize}
  \item 12\,=\,\textbf{ECC} (Ensemble of Classifier Chains): averages over
        multiple CCs with random chain orders.
\end{itemize}

"""

_METHOD_TAIL_ORIGINAL = r"""\paragraph{Reading the panels.} Each panel plots one dotted line with
markers per noise level, colored as in the original paper's Table~5:
\textcolor{red}{$\alpha=0.0$}, \textcolor{blue}{$\alpha=0.1$},
\textcolor{green!60!black}{$\alpha=0.2$}, \textcolor{cyan}{$\alpha=0.3$}.
\emph{PartialAbstention} and \emph{ScoreVector} panels show only indices
1--8 (the remaining baselines do not export a partial-abstention prediction
nor a probabilistic score vector). Vertical dashed lines at $x=4.5$,
$x=8.5$, and $x=11.5$ separate the four method groups: PA (1--4), PR
(5--8), standard baselines BR/CC/CLR (9--11), and ECC (12). Missing methods show as gaps in the line (with a small grey
``x'' on the axis when the method has no data at all for that panel).
"""

_METHOD_TAIL_ENHANCED = r"""\paragraph{Reading the panels.} Each panel plots one dotted line with
markers per noise level. \textbf{Noise is encoded by warm-hue intensity}
(ColorBrewer YlOrRd): yellow $=$ clean ($\alpha=0.0$), dark red $=$ noisy
($\alpha=0.3$):
\textcolor[HTML]{fecc5c}{$\blacksquare$}\,$\alpha=0.0$,
\textcolor[HTML]{fd8d3c}{$\blacksquare$}\,$\alpha=0.1$,
\textcolor[HTML]{f03b20}{$\blacksquare$}\,$\alpha=0.2$,
\textcolor[HTML]{bd0026}{$\blacksquare$}\,$\alpha=0.3$.
Below the numeric x-axis (1--12), a second tier prints the short method
name so the reader can identify each classifier without referring back to
this section. \emph{PartialAbstention} and \emph{ScoreVector} panels show
only indices 1--8. A vertical dashed line at $x=8.5$ separates the
order-based block (1--8) from the standard MLC baselines (9--12). The
y-axis is zoomed to the actual data range to keep close-clustered curves
visually separable. Missing methods show as gaps in the line (with a
small grey ``x'' on the axis when the method has no data at all for that
panel).
"""


def method_legend(style: str) -> str:
    tail = _METHOD_TAIL_ENHANCED if style == "enhanced" else _METHOD_TAIL_ORIGINAL
    return _METHOD_BLOCK_A_B + tail


def tex_escape(s: str) -> str:
    """Escape characters that confuse LaTeX in text mode."""
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("#", r"\#")


def list_metrics(fig_dir: Path) -> list[str]:
    return sorted(p.stem for p in fig_dir.glob("*.pdf"))


def emit_panel_table(
    rel_root: str, fig_dir: Path, caption: str, label: str, headline: str = ""
) -> str:
    """Build one \\begin{table}...\\end{table} block for a single (scope, ptype).

    ``headline`` (optional) is emitted as a non-numbered \\subsubsection*
    immediately before the table so the reader can scan section headers in
    the PDF outline / ToC without reading captions.
    """
    metrics = list_metrics(fig_dir)
    if not metrics:
        return f"% (no panels under {rel_root})\n"
    rows: list[str] = []
    for i in range(0, len(metrics), COLS_PER_ROW):
        cells = [
            f"\\mpanel{{{rel_root}/{m}.pdf}}" for m in metrics[i : i + COLS_PER_ROW]
        ]
        while len(cells) < COLS_PER_ROW:
            cells.append("")
        rows.append(" & ".join(cells) + r" \\")
    cols = "c" * COLS_PER_ROW
    head_block = f"\\subsubsection*{{{headline}}}\n" if headline else ""
    return (
        head_block
        + "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\setlength{\\tabcolsep}{2pt}\n"
        "\\renewcommand{\\arraystretch}{0.6}\n"
        f"\\begin{{tabular}}{{{cols}}}\n"
        + "\n".join(rows)
        + "\n\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def section_for(learner: str, scope: str, fig_root: Path) -> list[str]:
    """Emit table blocks for a learner-scope (scope in {'aggregate'} or 'per_dataset')."""
    out: list[str] = []
    if scope == "aggregate":
        for ptype_key in PTYPE_KEYS:
            fig_dir = fig_root / learner / "aggregate" / ptype_key
            if not fig_dir.exists():
                continue
            scope_word = "averaged across datasets"
            if learner == "lgbm":
                scope_word += " (8 of 9: enron LGBM still training)"
            cap = (
                f"{PTYPE_TITLE[ptype_key]} ({PTYPE_LABEL[ptype_key]}), "
                f"{learner.upper()} base learner, {scope_word}. "
                "Method indices and line colors follow the encoding paragraph above."
            )
            head = (
                f"{learner.upper()} $\\bullet$ Aggregate $\\bullet$ "
                f"{PTYPE_LABEL[ptype_key]} --- {PTYPE_TITLE[ptype_key]}"
            )
            out.append(
                emit_panel_table(
                    f"{learner}/aggregate/{ptype_key}",
                    fig_dir,
                    cap,
                    f"tab:{learner}_agg_{ptype_key}",
                    headline=head,
                )
            )
    elif scope == "per_dataset":
        for ds in DATASETS:
            ds_root = fig_root / learner / "per_dataset" / ds
            if not ds_root.exists():
                continue
            out.append(f"\\subsection{{{DATASET_DISPLAY.get(ds, ds)}}}\n")
            for ptype_key in PTYPE_KEYS:
                fig_dir = ds_root / ptype_key
                if not fig_dir.exists():
                    continue
                cap = (
                    f"{PTYPE_TITLE[ptype_key]} ({PTYPE_LABEL[ptype_key]}) on "
                    f"\\texttt{{{tex_escape(ds)}}} ({learner.upper()} base learner)."
                )
                head = (
                    f"{learner.upper()} $\\bullet$ \\texttt{{{tex_escape(ds)}}} "
                    f"$\\bullet$ {PTYPE_LABEL[ptype_key]} --- "
                    f"{PTYPE_TITLE[ptype_key]}"
                )
                out.append(
                    emit_panel_table(
                        f"{learner}/per_dataset/{ds}/{ptype_key}",
                        fig_dir,
                        cap,
                        f"tab:{learner}_{ds}_{ptype_key}",
                        headline=head,
                    )
                )
    return out


def emit_datasets_table(stats_path: Path) -> str:
    """Emit the datasets table that opens the experiments section.

    Reads paper_revision/dataset_stats.json (produced by
    compute_dataset_stats.py). Missing N values are rendered as ``?``.
    """
    if not stats_path.exists():
        return (
            "% dataset_stats.json missing — run "
            "scripts/paper_revision/compute_dataset_stats.py to populate it.\n"
        )
    stats: dict = json.loads(stats_path.read_text())
    if not stats:
        return "% dataset_stats.json empty.\n"

    order = DATASETS
    body = []
    for i, key in enumerate(order, start=1):
        if key not in stats:
            continue
        s = stats[key]
        n_str = "?" if s.get("N") is None else f"{s['N']}"
        body.append(
            f"{i} & {tex_escape(s['display'])} & {n_str} & {s['P']} & "
            f"{s['K']} & {s['MeanIR']:.2f} & {s['CVIR']:.2f} \\\\"
        )
    rows = "\n".join(body)
    return (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\begin{tabular}{l|l|r|r|r|r|r}\n"
        "\\hline\n"
        "\\# & Name & $N$ & $P$ & $K$ & MeanIR & CVIR \\\\\n"
        "\\hline\n"
        f"{rows}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Datasets used in the revised experiments. "
        "$N$, $P$, $K$ are the number of instances, features, labels. "
        "MeanIR and CVIR report per-label imbalance "
        "(higher = more imbalanced).}\n"
        "\\label{tab:datasets_revised}\n"
        "\\end{table}\n"
    )


PENDING_TODO = r"""\section*{Pending follow-ups (to fill before final submission)}
\begin{itemize}
  \item \textbf{ECC baseline on \texttt{birds} (RF + LGBM).} The
        \texttt{birds} folders contain only PA/PR/BR/CC/CLR --- ECC has not
        been run. Method slot~12 currently shows as a gap on every
        \texttt{birds} panel. Re-run with
        \texttt{scripts/train\_extra\_baselines.py --dataset birds
        --algorithm ecc} for both \texttt{--base\_learner rf} and
        \texttt{--base\_learner lgbm}.
  \item \textbf{LGBM run on \texttt{enron}.} Sequentially training (~2--3h
        per task due to K=53 pairwise classifiers); will be merged into
        \texttt{full\_enron\_split\_lgbm\_summary/} when complete and the
        LGBM aggregate panels regenerated.
  \item \textbf{Extended PartialAbstention metrics.} The evaluator has been
        extended with 11 new PA metrics
        (\texttt{jaccard\_pa}, \texttt{\{example,macro,micro\}\_\{precision,recall,f1\}\_pa},
        \texttt{mfrd\_pa}, \texttt{afrd\_pa}). Existing runs were summarised
        with the previous metric set; the new columns will populate
        automatically on the next end-to-end training+evaluation pass.
\end{itemize}

"""


_ABSTAIN_METRICS = ["f1_pa", "jaccard_pa"]


def emit_abstain_section(fig_root: Path) -> list[str]:
    """Emit the abstention-benefit section.

    Iterates over learners and emits one block per (learner, pa_metric)
    pair, where a block contains an aggregate figure plus a per-dataset
    figure for each dataset. The abstain charts live under
    ``<learner>/abstain/<pa_metric>/{aggregate,per_dataset/<ds>}/``.
    """
    parts: list[str] = []
    intro = (
        "\\section{Abstention vs standard MLC}\\label{sec:abstain_vs_mlc}\n"
        "\\paragraph{Why this section exists.} The paper's central claim is "
        "that an order-based predictor can \\emph{abstain} on labels it is "
        "unsure about and, on the labels it does predict, achieve higher "
        "quality than a standard MLC method that is forced to predict every "
        "label. The two chart families below make that claim visible.\n\n"
        "\\paragraph{Terminology.} `\\emph{Standard MLC}' (a.k.a.\\ "
        "\\texttt{BinaryVector} / BV) means each method outputs a 0/1 "
        "prediction for every one of the $K$ labels. `\\emph{With "
        "abstention}' (a.k.a.\\ \\texttt{PartialAbstention} / PA) means an "
        "order-based predictor may return $\\bot$ (abstain) on some labels; "
        "the PA quality metrics (\\texttt{f1\\_pa}, \\texttt{jaccard\\_pa}, "
        "\\ldots) are computed only over the non-abstained labels. Only the "
        "8 PA/PR methods (indices 1--8) can abstain; BR/CC/CLR/ECC cannot.\n\n"
        "\\paragraph{Reading the coverage-risk chart.} Each point is one "
        "(PA/PR method, noise level $\\alpha$) pair. The x-axis is the "
        "abstention rate (fraction of labels the method skipped); the "
        "y-axis is the quality metric on the labels that \\emph{were} "
        "predicted (higher is better). Marker shape encodes method, "
        "colour encodes noise level. The dashed grey line is the best "
        "standard-MLC baseline (BR / CC / CLR / ECC) on the BV quality "
        "metric --- it is the bar that ``no abstention'' has to clear. "
        "\\textbf{Any point above the dashed line is direct evidence that "
        "abstaining improved quality beyond what any standard MLC method "
        "achieves.} Points further right traded coverage for that gain.\n\n"
        "\\paragraph{Reading the paired-bars chart.} Same data, different "
        "view. The four sub-panels correspond to the four noise levels. "
        "Inside each sub-panel: 12 method positions, two bars per position "
        "--- blue is the standard-MLC quality (\\texttt{f1}, no "
        "abstention), warm-colour is the with-abstention quality on "
        "retained labels. Only the 8 PA/PR methods (1--8) have a warm "
        "bar; the standard MLC baselines (9--12) only have a blue bar by "
        "construction. The green ``$+x.y$'' label above each warm bar is "
        "the gain in score-points over that method's own BV bar.\n"
    )
    parts.append(intro)

    for learner_dir, learner_name in [("rf", "RF"), ("lgbm", "LGBM")]:
        learner_root = fig_root / learner_dir / "abstain"
        if not learner_root.exists():
            continue
        for pa_metric in _ABSTAIN_METRICS:
            metric_root = learner_root / pa_metric
            if not metric_root.exists():
                continue
            metric_tex = pa_metric.replace("_", r"\_")
            parts.append(
                f"\\subsection{{{learner_name}, metric \\texttt{{{metric_tex}}}"
                " --- aggregate}\n"
            )
            agg_dir = metric_root / "aggregate"
            if (agg_dir / "coverage_risk.pdf").exists():
                parts.append(
                    "\\begin{figure}[!htbp]\n\\centering\n"
                    f"\\mpanel{{{learner_dir}/abstain/{pa_metric}/aggregate/coverage_risk.pdf}}"
                    "\\hfill\n"
                    f"\\mpanel{{{learner_dir}/abstain/{pa_metric}/aggregate/paired_bars.pdf}}\n"
                    f"\\caption{{{learner_name} ({metric_tex}): coverage-risk"
                    " (left) and standard-MLC-vs-abstention paired bars"
                    " across $\\alpha\\in\\{0.0,0.1,0.2,0.3\\}$ (right),"
                    " averaged across all datasets.}\n"
                    f"\\label{{fig:abstain_{learner_dir}_{pa_metric}_agg}}\n"
                    "\\end{figure}\n"
                )
            per_ds_root = metric_root / "per_dataset"
            if per_ds_root.exists():
                parts.append(
                    f"\\subsection{{{learner_name}, metric \\texttt{{{metric_tex}}}"
                    " --- per dataset}\n"
                )
                for ds in DATASETS:
                    ds_dir = per_ds_root / ds
                    if not ds_dir.exists():
                        continue
                    parts.append(
                        "\\begin{figure}[!htbp]\n\\centering\n"
                        f"\\mpanel{{{learner_dir}/abstain/{pa_metric}/per_dataset/{ds}/coverage_risk.pdf}}"
                        "\\hfill\n"
                        f"\\mpanel{{{learner_dir}/abstain/{pa_metric}/per_dataset/{ds}/paired_bars.pdf}}\n"
                        f"\\caption{{{learner_name} on \\texttt{{{tex_escape(ds)}}}"
                        f" ({metric_tex}): coverage-risk (left) and"
                        " standard-MLC-vs-abstention paired bars (right).}\n"
                        f"\\label{{fig:abstain_{learner_dir}_{pa_metric}_{ds}}}\n"
                        "\\end{figure}\n"
                    )
            parts.append("\\clearpage\n")
    return parts


def build_tex(fig_root: Path, stats_path: Path, style: str) -> str:
    preamble = _make_preamble(fig_root.name, style)
    parts: list[str] = [preamble, "\\begin{document}\n\\maketitle\n"]

    parts.append("\\section{Datasets}\n")
    parts.append(emit_datasets_table(stats_path))

    parts.append(PENDING_TODO)

    parts.append(method_legend(style))

    parts.append("\\section{Main results: RF base learner, aggregated}\n")
    parts.extend(section_for("rf", "aggregate", fig_root))

    parts.append("\\clearpage\n")
    parts.extend(emit_abstain_section(fig_root))

    parts.append("\\clearpage\n\\appendix\n")
    parts.append("\\section{Per-dataset results (RF base learner)}\n")
    parts.extend(section_for("rf", "per_dataset", fig_root))

    parts.append("\\clearpage\n\\section{LGBM aggregated results}\n")
    parts.extend(section_for("lgbm", "aggregate", fig_root))

    parts.append("\n\\end{document}\n")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures_dir", default=str(DEFAULT_FIGURES))
    parser.add_argument("--stats", default=str(DEFAULT_STATS))
    parser.add_argument("--out", default=str(DEFAULT_TEX))
    parser.add_argument(
        "--style",
        default="original",
        choices=["original", "enhanced"],
        help=(
            "Visual style description in the method-encoding section. Must "
            "match the --style used when running build_panels.py so that the "
            "colour legend matches the rendered figures."
        ),
    )
    args = parser.parse_args()

    fig_root = Path(args.figures_dir)
    if not fig_root.exists():
        raise SystemExit(
            f"figures_dir {fig_root} does not exist. "
            "Run scripts/paper_revision/build_panels.py first."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_tex(fig_root, Path(args.stats), style=args.style))
    print(f"Wrote {out} (style={args.style})")


if __name__ == "__main__":
    main()
