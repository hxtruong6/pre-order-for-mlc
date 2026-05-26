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
    "CHD_49",
    "Water-quality",
    "enron",
]
DATASET_DISPLAY = {
    "GpositivePseAAC": "GpositivePseAAC ($K=4$)",
    "emotions": "Emotions ($K=6$)",
    "scene": "Scene ($K=6$)",
    "PlantPseAAC": "PlantPseAAC ($K=12$)",
    "HumanPseAAC": "HumanPseAAC ($K=14$)",
    "Yeast": "Yeast ($K=14$)",
    "CHD_49": "CHD\\_49 ($K=6$)",
    "Water-quality": "Water-quality ($K=14$)",
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
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{booktabs}
\usepackage{placeins}
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
\textcolor{green!60!black}{$\alpha=0.2$}, \textcolor{black}{$\alpha=0.3$}.
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
            # Force float queue to flush before the next dataset, so the
            # ~27 per-dataset tables don't all get pushed to the end where
            # they hit LaTeX's 18-float limit and get silently dropped.
            out.append("\\FloatBarrier\n")
    return out


# --- Paper-results section (mirrors the original paper's main tables) ----
# The original paper has three "main" tables in the Experiments section,
# each laid out as 3 datasets across, N metrics down:
#   - predict_binary_vector       (BV: 11 classifiers, MLC + ranking metrics)
#   - prediction_with_abstention  (PA: 8 classifiers, abstention metrics)
#   - score_vector ranking        (SV: 8 classifiers, probabilistic metrics)
# We reproduce this layout for the 9 revision datasets, grouped in triplets.
# Per the revision, f_Ham and f_sub are dropped in favour of f_Jaccard.
# (metric_key, math_label, ref_anchor, arrow). The bottom panel caption is
# composed in emit_paper_result_block as
# "{short_ds}: {math_label} (\ref{ref_anchor}) ({arrow})" — matches the
# original Table 4 format. The \ref anchors are intentional placeholders;
# when this block is pasted into the real paper, those refs resolve to the
# equation numbers automatically.
PAPER_TABLE_METRICS = {
    "bv": [
        ("f1",      r"$f^{1}_{\mathrm{MLC}}$",                 "eq:f1mlc",   r"$\uparrow$"),
        ("jaccard", r"$f^{\mathrm{Jacc}}_{\mathrm{MLC}}$",     "eq:jaccmlc", r"$\uparrow$"),
        ("afrd",    r"AFRD",                                   "eq:afrd",    r"$\downarrow$"),
        ("mfrd",    r"MFRD",                                   "eq:mfrd",    r"$\downarrow$"),
    ],
    # Per original paper Table 5: PA showed (f^1, f^ham, f^sub, AABS, ABS).
    # Revision drops ham/sub in favour of Jaccard but keeps the abstention
    # quantities AABS/ABS — they are essential characterisation of the PA
    # decision rule, not redundant with the score metrics.
    "pa": [
        ("f1_pa",      r"$f^{1}_{\mathrm{MLC}}$",             "eq:f1mlc",   r"$\uparrow$"),
        ("jaccard_pa", r"$f^{\mathrm{Jacc}}_{\mathrm{MLC}}$", "eq:jaccmlc", r"$\uparrow$"),
        ("aabs",       r"$\mathrm{AABS}$",                    "eq:aabs",    r"$\downarrow$"),
        ("abs",        r"$\mathrm{ABS}$",                     "eq:abs",     r"$\downarrow$"),
    ],
    "sv": [
        ("auc_macro",    r"AUROC",        "eq:auroc", r"$\uparrow$"),
        ("auprc_macro",  r"AUPRC",        "eq:auprc", r"$\uparrow$"),
        ("ranking_loss", r"Ranking loss", "eq:rloss", r"$\downarrow$"),
        ("one_error",    r"One-error",    "eq:oneerr", r"$\downarrow$"),
    ],
}
PAPER_DS_SHORT = {
    "GpositivePseAAC": "GpositivePse",
    "PlantPseAAC": "plantPse",
    "HumanPseAAC": "HumanPse",
    "emotions": "emotions",
    "scene": "scene",
    "Yeast": "Yeast",
    "CHD_49": "CHD-49",
    "Water-quality": "Water-quality",
    "enron": "enron",
}
# Group 1 matches the original paper's Table 4 datasets exactly
# (GpositivePse, PlantPse, HumanPse).
PAPER_TABLE_GROUPS = [
    ("GpositivePseAAC", "PlantPseAAC", "HumanPseAAC"),
]
PAPER_EXTRA_GROUPS = [
    ("emotions", "scene", "Yeast"),
    ("CHD_49", "Water-quality", "enron"),
]

# Full metric set for the appendix (Table G2/G3/G4). Does NOT drop ham/sub
# and adds macro-/micro-F1 for BV per the revision request.
# Matches the original paper's appendix split (imbalanced vs balanced
# datasets) — see the appendix_1/_2 labels in
# [Paper]…preference learning.tex. Original 8 datasets: VirusPse+CHD-49 in
# the imbalanced appendix; Emotion+Scene+Water-quality in the balanced one.
# In the revision, VirusPse is dropped and Yeast + enron are added; we
# keep the same convention — high-MeanIR datasets in the "imbalanced"
# bucket (CHD_49, Yeast, enron), balanced ones in the other.
PAPER_APPENDIX_DATASETS = {
    "imbalanced": ("CHD_49", "Yeast", "enron"),
    "balanced":   ("emotions", "scene", "Water-quality"),
}
PAPER_APPENDIX_METRICS = {
    # Original BV appendix metric set (Tables G2, G3 in the original paper).
    "bv_orig": [
        ("f1",               r"$f^{1}_{\mathrm{MLC}}$",            "eq:f1mlc",  r"$\uparrow$"),
        ("hamming_accuracy", r"$f^{\mathrm{ham}}_{\mathrm{MLC}}$", "eq:fham",   r"$\uparrow$"),
        ("subset0_1",        r"$f^{\mathrm{sub}}_{\mathrm{MLC}}$", "eq:fsub",   r"$\uparrow$"),
        ("afrd",             r"AFRD",                              "eq:afrd",   r"$\downarrow$"),
        ("mfrd",             r"MFRD",                              "eq:mfrd",   r"$\downarrow$"),
    ],
    # Original PA appendix metric set (Tables G4, G5 in the original paper).
    "pa_orig": [
        ("f1_pa",               r"$f^{1}_{\mathrm{MLC}}$",            "eq:f1mlc", r"$\uparrow$"),
        ("hamming_accuracy_pa", r"$f^{\mathrm{ham}}_{\mathrm{MLC}}$", "eq:fham",  r"$\uparrow$"),
        ("subset0_1_pa",        r"$f^{\mathrm{sub}}_{\mathrm{MLC}}$", "eq:fsub",  r"$\uparrow$"),
        ("aabs",                r"AABS",                              "eq:aabs",  r"$\downarrow$"),
        ("abs",                 r"ABS",                               "eq:abs",   r"$\downarrow$"),
    ],
    # New metrics added in the revision — jaccard + macro/micro-F1.
    "bv_new": [
        ("jaccard",  r"$f^{\mathrm{Jacc}}_{\mathrm{MLC}}$", "eq:jaccmlc", r"$\uparrow$"),
        ("macro_f1", r"macro-$f_1$",                        "eq:macrof1", r"$\uparrow$"),
        ("micro_f1", r"micro-$f_1$",                        "eq:microf1", r"$\uparrow$"),
    ],
    "pa_new": [
        ("jaccard_pa",  r"$f^{\mathrm{Jacc}}_{\mathrm{MLC}}$", "eq:jaccmlc", r"$\uparrow$"),
        ("macro_f1_pa", r"macro-$f_1$",                        "eq:macrof1", r"$\uparrow$"),
        ("micro_f1_pa", r"micro-$f_1$",                        "eq:microf1", r"$\uparrow$"),
    ],
    # Full SV metric set (revision-only; original paper had no SV table).
    "sv": [
        ("auc_macro",    r"AUROC (macro)", "eq:aurocM", r"$\uparrow$"),
        ("auc_micro",    r"AUROC (micro)", "eq:aurocm", r"$\uparrow$"),
        ("auprc_macro",  r"AUPRC (macro)", "eq:auprcM", r"$\uparrow$"),
        ("auprc_micro",  r"AUPRC (micro)", "eq:auprcm", r"$\uparrow$"),
        ("ranking_loss", r"Ranking loss",  "eq:rloss",  r"$\downarrow$"),
        ("one_error",    r"One-error",     "eq:oneerr", r"$\downarrow$"),
        ("coverage",     r"Coverage",      "eq:cov",    r"$\downarrow$"),
        ("lr_ap",        r"LR-AP",         "eq:lrap",   r"$\uparrow$"),
    ],
}
MAIN_DATASETS = ("GpositivePseAAC", "PlantPseAAC", "HumanPseAAC")
# Each entry: (caption_tag, panel_dir, datasets, metrics_key, label_id)
PAPER_APPENDIX_BLOCKS = [
    ("(Table G2)", "bv", PAPER_APPENDIX_DATASETS["imbalanced"], "bv_orig", "g2"),
    ("(Table G3)", "bv", PAPER_APPENDIX_DATASETS["balanced"],   "bv_orig", "g3"),
    ("(Table G4)", "pa", PAPER_APPENDIX_DATASETS["imbalanced"], "pa_orig", "g4"),
    ("(Table G5)", "pa", PAPER_APPENDIX_DATASETS["balanced"],   "pa_orig", "g5"),
    # G6: new BV metrics (jaccard + macro/micro-F1) across all 9 datasets.
    ("(Table G6, new metrics --- main datasets)", "bv", MAIN_DATASETS,                                "bv_new", "g6a"),
    ("(Table G6, new metrics --- imbalanced)",    "bv", PAPER_APPENDIX_DATASETS["imbalanced"],        "bv_new", "g6b"),
    ("(Table G6, new metrics --- balanced)",      "bv", PAPER_APPENDIX_DATASETS["balanced"],          "bv_new", "g6c"),
    ("(Table G7, new metrics --- main datasets)", "pa", MAIN_DATASETS,                                "pa_new", "g7a"),
    ("(Table G7, new metrics --- imbalanced)",    "pa", PAPER_APPENDIX_DATASETS["imbalanced"],        "pa_new", "g7b"),
    ("(Table G7, new metrics --- balanced)",      "pa", PAPER_APPENDIX_DATASETS["balanced"],          "pa_new", "g7c"),
    # G8 (ScoreVector) removed per user request — the SV view did not
    # surface useful information beyond what AUROC/AUPRC already convey
    # in the per-dataset appendix.
]
PAPER_PANEL_WIDTH = r"0.30\textwidth"
PAPER_CELL_WIDTH = r"0.32\textwidth"
# Per-style hex colors for the noise levels, mirroring NOISE_COLORS_*
# in build_panels.py. Used to render the coloured α tokens in captions
# so the legend matches the markers in each subpanel.
PAPER_NOISE_COLORS = {
    "original": {
        "0.0": "FF0000",  # red
        "0.1": "0000FF",  # blue
        "0.2": "339933",  # green
        "0.3": "000000",  # black
    },
    "enhanced": {
        "0.0": "fecc5c",
        "0.1": "fd8d3c",
        "0.2": "f03b20",
        "0.3": "bd0026",
    },
}


def _paper_noise_legend_tex(style: str) -> str:
    """Return the coloured `α = 0.0, α = 0.1, α = 0.2 and α = 0.3` snippet."""
    cols = PAPER_NOISE_COLORS.get(style, PAPER_NOISE_COLORS["original"])
    tokens = [
        f"\\textcolor[HTML]{{{cols['0.0']}}}{{$\\alpha = 0.0$}}",
        f"\\textcolor[HTML]{{{cols['0.1']}}}{{$\\alpha = 0.1$}}",
        f"\\textcolor[HTML]{{{cols['0.2']}}}{{$\\alpha = 0.2$}}",
        f"\\textcolor[HTML]{{{cols['0.3']}}}{{$\\alpha = 0.3$}}",
    ]
    return ", ".join(tokens[:3]) + " and " + tokens[3]


# Tag mapping back to the original paper so the reader (and you, when
# copy-pasting) can locate which original table each block replaces.
PAPER_TABLE_ORIG_TAG = {
    "bv": "(Table 4)",
    "pa": "(Table 5)",
    # SV has no counterpart in the original paper — it is new in the revision.
    "sv": "(new)",
}
PAPER_CAPTION_BY_PTYPE = {
    "bv": (
        "Average scores (in \\%, $y$-axis) over $5\\times 5$ cross-validation "
        "folds, plotted against the encoded number of the classifiers "
        "($x$-axis): (PA-H-2, PA-H, PA-S-2, PA-S), "
        "(PR-H-2, PR-H, PR-S-2, PR-S), and (BR, CC, CLR) are encoded as "
        "(1, 2, 3, 4), (5, 6, 7, 8), and (9, 10, 11), respectively. "
        "Results are color-coded with respect to the noisy level $\\alpha$ "
        "as follows: __NOISE_LEGEND__."
    ),
    "pa": (
        "Average scores (in \\%, $y$-axis) over $5\\times 5$ cross-validation "
        "folds, plotted against the encoded number of the classifiers "
        "($x$-axis): (PA-H-2, PA-H, PA-S-2, PA-S) and "
        "(PR-H-2, PR-H, PR-S-2, PR-S) are encoded as (1, 2, 3, 4) and "
        "(5, 6, 7, 8), respectively. Results are color-coded with respect "
        "to the noisy level $\\alpha$ as follows: __NOISE_LEGEND__."
    ),
    "sv": (
        "Average scores (in \\%, $y$-axis) over $5\\times 5$ cross-validation "
        "folds, plotted against the encoded number of the classifiers "
        "($x$-axis): (PA-H-2, PA-H, PA-S-2, PA-S) and "
        "(PR-H-2, PR-H, PR-S-2, PR-S) are encoded as (1, 2, 3, 4) and "
        "(5, 6, 7, 8), respectively. Results are color-coded with respect "
        "to the noisy level $\\alpha$ as follows: __NOISE_LEGEND__."
    ),
}


def emit_paper_result_block(
    learner: str,
    ptype: str,
    datasets: tuple[str, str, str],
    fig_root: Path,
    group_idx: int,
    style: str = "original",
    metrics: list | None = None,
    caption_tag: str | None = None,
    label_prefix: str = "paper",
    cell_width: str | None = None,
    array_stretch: str | None = None,
) -> str:
    """Emit one Table-4-style block: 3 datasets across × N metrics down."""
    if metrics is None:
        metrics = PAPER_TABLE_METRICS[ptype]
    if caption_tag is None:
        caption_tag = PAPER_TABLE_ORIG_TAG[ptype]
    # Auto-shrink tall blocks (≥7 metrics × 3 datasets) so they fit one
    # page. Picked to keep G8 (SV full, 8 metrics) and G2-full (8 metrics
    # variant) within the page text-height budget.
    n_metrics = len(metrics)
    eff_cell_width = cell_width or (
        r"0.27\textwidth" if n_metrics >= 7 else PAPER_CELL_WIDTH
    )
    eff_array_stretch = array_stretch or ("0.65" if n_metrics >= 7 else "0.8")
    rows: list[str] = []
    header_cells = [
        f"\\textbf{{{PAPER_DS_SHORT.get(ds, ds)}}}" for ds in datasets
    ]
    rows.append(" & ".join(header_cells) + r" \\")
    for metric_key, math_label, eq_ref, arrow in metrics:
        cells = []
        for ds in datasets:
            pdf_rel = f"{learner}/per_dataset/{ds}/{ptype}/{metric_key}.pdf"
            full = fig_root / pdf_rel
            if full.exists():
                caption = (
                    f"\\footnotesize {math_label} "
                    f"(\\ref{{{eq_ref}}}) ({arrow})"
                )
                # trim=l b r t (bp). Crop ~14bp off the top so the
                # per-panel matplotlib title ("f1", "jaccard", ...) is
                # not visible — the bottom LaTeX caption already labels
                # this panel. The appendix tables (Table 12+) reuse the
                # uncropped PDFs and keep the title for tracking.
                cells.append(
                    f"\\begin{{minipage}}{{{eff_cell_width}}}\\centering"
                    f"\\includegraphics[width=\\linewidth,"
                    f"trim=0 0 0 14, clip]{{{pdf_rel}}}\\\\"
                    f"{caption}"
                    f"\\end{{minipage}}"
                )
            else:
                cells.append(f"\\footnotesize (missing: {metric_key})")
        rows.append(" & ".join(cells) + r" \\")
    cols = "c" * 3
    ds_labels = ", ".join(datasets)
    bookmark_title = f"Group {group_idx}: {ds_labels}"
    bookmark_anchor = f"{label_prefix}-{learner}-{ptype}-g{group_idx}"
    return (
        f"\\pdfbookmark[3]{{{caption_tag} {bookmark_title}}}{{{bookmark_anchor}}}\n"
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\setlength{\\tabcolsep}{2pt}\n"
        f"\\renewcommand{{\\arraystretch}}{{{eff_array_stretch}}}\n"
        f"\\begin{{tabular}}{{{cols}}}\n"
        + "\n".join(rows)
        + "\n\\end{tabular}\n"
        f"\\caption{{{caption_tag} "
        f"{PAPER_CAPTION_BY_PTYPE[ptype].replace('__NOISE_LEGEND__', _paper_noise_legend_tex(style))}}}\n"
        f"\\label{{tab:{label_prefix}_{learner}_{ptype}_g{group_idx}}}\n"
        "\\end{table}\n"
    )


def emit_paper_results_section(
    learner: str,
    fig_root: Path,
    stats_path: Path | None = None,
    style: str = "original",
) -> list[str]:
    """Section mirroring the original paper's main Experiments tables."""
    out = [
        f"\\section{{Paper results ({learner.upper()} base learner)}}\n",
        "Reproduces the layout of the original paper's main experiment "
        "tables (Tables~5--7): 3 datasets across, 4 metrics down, one "
        "table per prediction type (binary vector, partial abstention, "
        "score vector). Per the revision, $f^{\\mathrm{ham}}_{\\mathrm{MLC}}$ "
        "and $f^{\\mathrm{sub}}_{\\mathrm{MLC}}$ are dropped in favour of "
        "$f^{\\mathrm{Jacc}}_{\\mathrm{MLC}}$.\n\n",
    ]
    if stats_path is not None:
        out.append(
            "\\pdfbookmark[2]{Table 3: Datasets}{paper-"
            f"{learner}-datasets" "}\n"
        )
        out.append(emit_datasets_table(stats_path))
        out.append("\\FloatBarrier\n")
    # SV (ScoreVector) removed per user request — the SV main-paper view
    # added no clear signal beyond what BV + PA already show.
    for ptype in ("bv", "pa"):
        out.append(
            f"\\subsection{{{PTYPE_TITLE[ptype]} ({PTYPE_LABEL[ptype]})}}\n"
        )
        for idx, group in enumerate(PAPER_TABLE_GROUPS, start=1):
            out.append(
                emit_paper_result_block(learner, ptype, group, fig_root, idx, style)
            )
            out.append("\\FloatBarrier\n")
    return out


def emit_paper_appendix_section(
    learner: str,
    fig_root: Path,
    style: str = "original",
    as_subsection: bool = False,
) -> list[str]:
    """Appendix tables G2..G7 mirroring the original paper's appendix.

    Original paper convention:
      G2/G3 — BV results on imbalanced / balanced appendix datasets.
      G4/G5 — PA results on imbalanced / balanced appendix datasets.
    Revision additions:
      G6/G7 — new BV/PA metrics (jaccard, macro-/micro-F1) on all 9 datasets.
    """
    top = "\\subsection" if as_subsection else "\\section"
    sub = "\\subsubsection" if as_subsection else "\\subsection"
    out = [
        f"{top}{{Experimental results --- appendix tables "
        f"({learner.upper()} base learner)}}"
        f"\\label{{sec:paperapp-{learner}}}\n",
        "Appendix counterpart of the main Paper results section. Tables "
        "G2--G5 mirror the original paper's appendix exactly (same metric "
        "sets, with the 6 non-main datasets split into imbalanced "
        "vs.\\ balanced buckets following the original layout). Tables "
        "G6--G7 are revision additions reporting the new metrics "
        "($f^{\\mathrm{Jacc}}_{\\mathrm{MLC}}$, macro-$f_1$, micro-$f_1$).\n\n",
    ]
    # Iterate the original-paper-style appendix blocks G2..G8.
    last_panel_dir = None
    for caption_tag, panel_dir, datasets, mkey, label_id in PAPER_APPENDIX_BLOCKS:
        if panel_dir != last_panel_dir:
            out.append(
                f"{sub}{{{PTYPE_TITLE[panel_dir]} "
                f"({PTYPE_LABEL[panel_dir]})}}\n"
            )
            last_panel_dir = panel_dir
        out.append(
            emit_paper_result_block(
                learner,
                panel_dir,
                datasets,
                fig_root,
                1,  # single group per block
                style,
                metrics=PAPER_APPENDIX_METRICS[mkey],
                caption_tag=caption_tag,
                label_prefix=f"paperapp_{label_id}",
            )
        )
        out.append("\\FloatBarrier\n")
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

    # Sort by K, then N, then P (ascending) for the table; original
    # DATASETS order is preserved everywhere else (figure sections, etc.).
    order = sorted(
        (k for k in DATASETS if k in stats),
        key=lambda k: (
            stats[k].get("K", 0),
            stats[k].get("N", 0) or 0,
            stats[k].get("P", 0) or 0,
        ),
    )
    body = []
    for i, key in enumerate(order, start=1):
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
        "\\caption{(Table 3) Datasets used in the revised experiments. "
        "$N$, $P$, $K$ are the number of instances, features, labels. "
        "MeanIR and CVIR report per-label imbalance "
        "(higher = more imbalanced).}\n"
        "\\label{tab:datasets_revised}\n"
        "\\end{table}\n"
    )


PENDING_TODO = r"""\section*{Pending follow-ups (to fill before final submission)}
\begin{itemize}
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


def _abstain_figure_block(
    fig_root: Path,
    learner_dir: str,
    learner_name: str,
    pa_metric: str,
    metric_tex: str,
    scope_dir: str,
    scope_label: str,
    label_suffix: str,
) -> list[str]:
    """Emit three figures (existing pair, gain pair, robustness pair) for a scope."""
    base = f"{learner_dir}/abstain/{pa_metric}/{scope_dir}"
    abs_path = fig_root / learner_dir / "abstain" / pa_metric / scope_dir
    if not abs_path.exists():
        return []

    # Shared (metric-independent) abstention-rate chart sits next to the
    # quality chart. Path mirrors the scope_dir but lives under _shared/.
    shared_base = f"{learner_dir}/abstain/_shared/{scope_dir}"

    out: list[str] = []
    # Figure 1a: abstention rates (left) next to paired_bars quality (right).
    out.append(
        "\\begin{figure}[!htbp]\n\\centering\n"
        f"\\mpanel{{{shared_base}/abstention_rate.pdf}}\\hfill\n"
        f"\\mpanel{{{base}/paired_bars.pdf}}\n"
        f"\\caption{{{learner_name} ({metric_tex}) {scope_label}: "
        "abstention rates (left) and standard-MLC-vs-abstention paired bars "
        "(right), across $\\alpha\\in\\{0.0,0.1,0.2,0.3\\}$. "
        "Left: bar = \\texttt{abs} (\\% of instances with at least one "
        "abstained label, instance-level); dot = \\texttt{aabs} (\\% of "
        "labels skipped, per-cell label-level). These rates are independent "
        "of the quality metric. Right: blue = standard MLC quality (no "
        "abstention) on the matched BV metric; coloured = with-abstention "
        "quality on retained labels.}\n"
        f"\\label{{fig:abstain_{learner_dir}_{pa_metric}_{label_suffix}_rates_quality}}\n"
        "\\end{figure}\n"
    )
    # Figure 1b: coverage-risk scatter (standalone).
    out.append(
        "\\begin{figure}[!htbp]\n\\centering\n"
        f"\\mpanel{{{base}/coverage_risk.pdf}}\n"
        f"\\caption{{{learner_name} ({metric_tex}) {scope_label}: "
        "coverage-risk scatter. Each point is one (method, noise) pair; "
        "x = abstention rate, y = quality on retained labels. Points above "
        "the dashed line beat the best standard-MLC baseline.}\n"
        f"\\label{{fig:abstain_{learner_dir}_{pa_metric}_{label_suffix}_coverage_risk}}\n"
        "\\end{figure}\n"
    )
    # Figure 2: Pareto + heatmap.
    out.append(
        "\\begin{figure}[!htbp]\n\\centering\n"
        f"\\mpanel{{{base}/coverage_risk_pareto.pdf}}\\hfill\n"
        f"\\mpanel{{{base}/gain_heatmap.pdf}}\n"
        f"\\caption{{{learner_name} ({metric_tex}) {scope_label}:"
        f" Pareto frontier of (abstention, retained quality) with the"
        f" win-region shaded green (left); per-method per-noise gain"
        f" $(\\text{{{metric_tex}}}-f_1)$ heatmap (right).}}\n"
        f"\\label{{fig:abstain_{learner_dir}_{pa_metric}_{label_suffix}_gain}}\n"
        "\\end{figure}\n"
    )
    # Figure 3: robustness (efficiency-quality + effective f1).
    out.append(
        "\\begin{figure}[!htbp]\n\\centering\n"
        f"\\mpanel{{{base}/efficiency_quality.pdf}}\\hfill\n"
        f"\\mpanel{{{base}/effective_f1.pdf}}\n"
        f"\\caption{{{learner_name} ({metric_tex}) {scope_label} ---"
        f" robustness view. Left: efficiency-quality scatter"
        f" (x = coverage $= 1-\\text{{abs}}$); upper-right is robust"
        f" (predicts many labels and keeps quality high). Right:"
        f" effective {metric_tex} $= (1-\\text{{abs}}) \\cdot {metric_tex}$,"
        f" i.e.\\ retained quality penalised by skipped coverage. A"
        f" method whose effective curve stays above the dashed standard-MLC"
        f" baseline at every $\\alpha$ is robust to noise, not just"
        f" selectively abstaining on hard labels.}}\n"
        f"\\label{{fig:abstain_{learner_dir}_{pa_metric}_{label_suffix}_robust}}\n"
        "\\end{figure}\n"
    )
    out.append("\\FloatBarrier\n")
    return out


def emit_abstain_overview(
    fig_root: Path, as_subsection: bool = False
) -> list[str]:
    """Compact overview: 4 aggregate paired_bars (RF/LGBM x f1/jaccard).

    One figure per (learner, pa_metric) so the reader can scan the headline
    trend in <1 page before opening the detailed per-method / per-dataset
    section.
    """
    top = "\\subsection" if as_subsection else "\\section"
    parts: list[str] = [
        f"{top}{{Abstention overview: aggregate trends}}"
        "\\label{sec:abstain_overview}\n"
        "\\paragraph{Purpose.} This compact section shows the headline "
        "abstention-vs-standard-MLC trend across noise levels, averaged over "
        "all datasets, for both base learners (RF, LGBM) and both retained "
        "quality metrics ($f_1$, Jaccard). Use it as a quick visual summary "
        "before diving into the detailed per-method / per-dataset figures in "
        "Section~\\ref{sec:abstain_vs_mlc}.\n\n"
        "\\paragraph{How to read.} In each panel: blue bar = standard MLC "
        "(no abstention) on its native quality metric; coloured bar = "
        "with-abstention quality on retained labels. Green ``$+x.y$'' label "
        "= score-point gain of abstention over the standard-MLC bar of the "
        "same method. The four sub-panels per figure correspond to "
        "$\\alpha\\in\\{0.0, 0.1, 0.2, 0.3\\}$.\n"
    ]
    # Only the table-shaped 4x3 grid is emitted in the paper-ready section.
    # The other layouts (1.1 paired bars, 1.2 gain-only, 1.3 2-row) are
    # still generated by build_abstain_charts.py for ad-hoc inspection but
    # do not appear in the final PDF.
    figure_specs = [
        ("summary_grid.pdf", "1",
         "table-shaped $4\\times 3$ grid: rows $=\\alpha\\in"
         "\\{0.0, 0.1, 0.2, 0.3\\}$, columns $=$ ((a) $f_1$ paired bars, "
         "(b) Jaccard paired bars, (c) abstention rate). Light blue $=$ "
         "standard MLC, $\\alpha$-coloured $=$ with-abstention; in (c), "
         "bar $=$ \\texttt{abs} (instance-level), dot $=$ \\texttt{aabs} "
         "(per-cell)."),
    ]
    for learner_dir, learner_name in [("rf", "RF"), ("lgbm", "LGBM")]:
        for fname, fig_idx, caption_detail in figure_specs:
            chart = fig_root / learner_dir / "abstain" / "_shared" / "aggregate" / fname
            if not chart.exists():
                continue
            rel = f"{learner_dir}/abstain/_shared/aggregate/{fname}"
            parts.append(
                "\\begin{figure}[!htbp]\n\\centering\n"
                f"\\includegraphics[width=\\linewidth]{{{rel}}}\n"
                f"\\caption{{Figure~{fig_idx}: {learner_name} "
                f"averaged across all datasets --- {caption_detail} "
                "Color = $\\alpha\\in\\{0.0, 0.1, 0.2, 0.3\\}$.}\n"
                f"\\label{{fig:abstain_summary_{learner_dir}_{fig_idx.replace('.', '_')}}}\n"
                "\\end{figure}\n"
            )
    parts.append("\\FloatBarrier\n\\clearpage\n")
    return parts


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
            parts.extend(_abstain_figure_block(
                fig_root, learner_dir, learner_name, pa_metric, metric_tex,
                scope_dir="aggregate", scope_label="averaged across all datasets",
                label_suffix="agg",
            ))
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
                    parts.extend(_abstain_figure_block(
                        fig_root, learner_dir, learner_name, pa_metric, metric_tex,
                        scope_dir=f"per_dataset/{ds}",
                        scope_label=f"on \\texttt{{{tex_escape(ds)}}}",
                        label_suffix=ds,
                    ))
            parts.append("\\clearpage\n")
    return parts


def build_tex(fig_root: Path, stats_path: Path, style: str) -> str:
    preamble = _make_preamble(fig_root.name, style)
    parts: list[str] = [preamble, "\\begin{document}\n\\maketitle\n"]

    parts.append("\\section{Datasets}\n")
    parts.append(emit_datasets_table(stats_path))

    parts.append(PENDING_TODO)

    parts.append(method_legend(style))

    # §3 Paper results — single section containing everything paper-ready:
    # Tables 3/4/5/SV-new + appendix tables G2..G8 + abstention overview
    # figures. User wants this whole block under one section heading so
    # they can open and copy-paste straight into the original paper.
    parts.append("\\clearpage\n")
    parts.extend(emit_paper_results_section("rf", fig_root, stats_path, style))
    parts.extend(emit_paper_appendix_section("rf", fig_root, style,
                                              as_subsection=True))
    parts.extend(emit_abstain_overview(fig_root, as_subsection=True))

    # Detailed supporting sections (not for paper copy-paste).
    parts.append("\\clearpage\n")
    parts.append("\\section{Main results: RF base learner, aggregated}\n")
    parts.extend(section_for("rf", "aggregate", fig_root))

    parts.append("\\clearpage\n")
    parts.extend(emit_abstain_section(fig_root))

    parts.append("\\clearpage\n\\appendix\n")
    parts.extend(emit_paper_appendix_section("rf", fig_root, style))
    parts.append("\\clearpage\n")
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
