"""
Generate the v2 result figure suite for the preorder4MLC paper.

Run:
    python utils/plot_figures.py \
        --results_dir results/final_20260514_v2_summary \
        --raw_results_dir results/20260514_v2

Outputs land in ``<results_dir>/figures/`` and are split into subfolders:

    cd_diagrams/        Demsar-style critical-difference diagrams
    noise_curves/       Average rank vs noise rate (one line per algorithm)
    abstention/         Abstention-frontier scatter plots
    tradeoff/           Hamming vs Subset accuracy trade-off
    heatmaps/           Avg-rank heatmap and best-variant-per-dataset map

Each figure is saved as ``.pdf`` (publication), ``.png`` (preview at 150 dpi),
and ``.tex`` via ``matplot2tikz`` when possible.  TikZ export failures are
logged and skipped without aborting the whole run.

This script reads the per-dataset ``*_summary.csv`` files written by
``utils/summarize_metrics.py`` plus the average-rank CSVs written by
``utils/statistical_tests.py``; it does not touch the raw pickle results
under ``--raw_results_dir`` (the argument is kept for future use / logging).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare, rankdata, studentized_range

try:
    import matplot2tikz
except Exception as exc:  # pragma: no cover - import guard
    matplot2tikz = None
    print(f"[warn] matplot2tikz unavailable: {exc}")


# ---------------------------------------------------------------------------
# Reuse the metric direction constants from statistical_tests.py as the
# authoritative source.  We import lazily so this script remains usable when
# the package layout changes; we fall back to inlined copies otherwise.
# ---------------------------------------------------------------------------

try:
    from utils.statistical_tests import (  # type: ignore
        HIGHER_IS_BETTER,
        LOWER_IS_BETTER,
        parse_mean,
        metric_direction,
    )
except Exception:  # pragma: no cover - fallback when run as a loose script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from statistical_tests import (  # type: ignore
        HIGHER_IS_BETTER,
        LOWER_IS_BETTER,
        parse_mean,
        metric_direction,
    )


# ---------------------------------------------------------------------------
# Constants & display helpers
# ---------------------------------------------------------------------------

# IA1-IA8 ordering, copied verbatim from
# evaluation_test.py::EvaluationConfig.INFERENCE_ALGORITHMS so the figures stay
# in sync with the rest of the paper.
IA_ORDER: List[Tuple[str, str]] = [
    ("IA1", "PreOrder__Hamming__None"),
    ("IA2", "PreOrder__Hamming__2"),
    ("IA3", "PreOrder__Subset__None"),
    ("IA4", "PreOrder__Subset__2"),
    ("IA5", "PartialOrder__Hamming__None"),
    ("IA6", "PartialOrder__Hamming__2"),
    ("IA7", "PartialOrder__Subset__None"),
    ("IA8", "PartialOrder__Subset__2"),
]
ALG_TO_IA: Dict[str, str] = {full: short for short, full in IA_ORDER}
IA_TO_ALG: Dict[str, str] = {short: full for short, full in IA_ORDER}
IA_SHORT_NAMES: List[str] = [short for short, _ in IA_ORDER]

BASELINE_DISPLAY: Dict[str, str] = {
    "br": "BR",
    "cc": "CC",
    "clr": "CLR",
    "mlknn": "MLkNN",
    "ecc": "ECC",
    "lp": "LP",
}
BASELINES_ORDER: List[str] = ["br", "cc", "clr", "mlknn", "ecc", "lp"]

NOISE_LEVELS: List[str] = ["0.0", "0.1", "0.2", "0.3"]

DATASETS: List[str] = [
    "CHD_49",
    "emotions",
    "scene",
    "VirusPseAAC",
    "Yeast",
    "Water-quality",
    "HumanPseAAC",
    "GpositivePseAAC",
    "PlantPseAAC",
]


def display_name(alg: str) -> str:
    if alg in ALG_TO_IA:
        return ALG_TO_IA[alg]
    if alg in BASELINE_DISPLAY:
        return BASELINE_DISPLAY[alg]
    return alg


def is_bopos(alg: str) -> bool:
    return alg in ALG_TO_IA


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SUMMARY_RE = re.compile(
    r"^(?P<dataset>.+)_(?P<ptype>BinaryVector|PartialAbstention|ScoreVector)_summary\.csv$"
)


def load_summary(path: str) -> pd.DataFrame:
    """Return a long DataFrame with columns: algorithm, metric, noise, mean."""
    df = pd.read_csv(path)
    if "Algorithm" not in df.columns:
        return pd.DataFrame(columns=["algorithm", "metric", "noise", "mean"])
    rows = []
    for _, row in df.iterrows():
        alg = str(row["Algorithm"]).strip()
        if not alg or alg.lower() == "nan":
            continue
        for col in df.columns:
            if col == "Algorithm" or "__" not in col:
                continue
            metric, noise = col.rsplit("__", 1)
            rows.append((alg, metric, noise, parse_mean(row[col])))
    return pd.DataFrame(rows, columns=["algorithm", "metric", "noise", "mean"])


def load_all_summaries(results_dir: str) -> pd.DataFrame:
    """Long DataFrame with columns: dataset, ptype, algorithm, metric, noise, mean."""
    rows = []
    for path in sorted(glob(os.path.join(results_dir, "*_summary.csv"))):
        m = SUMMARY_RE.match(os.path.basename(path))
        if not m:
            continue
        df = load_summary(path)
        df["dataset"] = m.group("dataset")
        df["ptype"] = m.group("ptype")
        rows.append(df)
    if not rows:
        raise SystemExit(f"No *_summary.csv files found in {results_dir}")
    return pd.concat(rows, ignore_index=True)


def pivot_metric(
    long_df: pd.DataFrame,
    ptype: str,
    metric: str,
    noise: str,
) -> pd.DataFrame:
    """Return a [dataset x algorithm] matrix of means."""
    sub = long_df[
        (long_df["ptype"] == ptype)
        & (long_df["metric"] == metric)
        & (long_df["noise"] == noise)
    ]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="dataset", columns="algorithm", values="mean", aggfunc="mean")


def load_significance_summary(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "stats", "significance_summary.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["noise"] = df["noise"].astype(str)
    return df


def friedman_p(sig_df: pd.DataFrame, ptype: str, metric: str, noise: str) -> Optional[float]:
    if sig_df.empty:
        return None
    row = sig_df[
        (sig_df["ptype"] == ptype)
        & (sig_df["metric"] == metric)
        & (sig_df["noise"] == str(noise))
    ]
    if row.empty:
        return None
    return float(row.iloc[0]["friedman_p"])


# ---------------------------------------------------------------------------
# Saving helper
# ---------------------------------------------------------------------------


def save_figure(fig: plt.Figure, out_dir: str, name: str) -> None:
    """Save ``fig`` as PDF + PNG + (optionally) TikZ."""
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    png_path = os.path.join(out_dir, f"{name}.png")
    tex_path = os.path.join(out_dir, f"{name}.tex")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    if matplot2tikz is not None:
        try:
            matplot2tikz.save(tex_path, figure=fig, strict=False)
        except Exception as exc:
            print(f"[warn] matplot2tikz failed for {name}: {exc}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1) CD diagrams
# ---------------------------------------------------------------------------


def rank_matrix(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    arr = -values if higher_is_better else values
    ranks = np.empty_like(arr, dtype=float)
    for i in range(arr.shape[0]):
        row = arr[i]
        mask = ~np.isnan(row)
        ranks[i, :] = np.nan
        if mask.sum() == 0:
            continue
        ranks[i, mask] = rankdata(row[mask], method="average")
    return ranks


def critical_difference(k: int, n_datasets: int, alpha: float = 0.05) -> float:
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_datasets))


def plot_cd_diagram(
    avg_ranks: np.ndarray,
    names: Sequence[str],
    cd: float,
    title: str,
    out_dir: str,
    name: str,
    friedman_p_value: Optional[float] = None,
) -> None:
    k = len(names)
    order = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[order]
    sorted_names = [names[i] for i in order]

    lo = int(np.floor(sorted_ranks.min()))
    hi = int(np.ceil(sorted_ranks.max()))
    if hi == lo:
        hi = lo + 1

    fig_w = max(8.0, 1.1 * k)
    fig, ax = plt.subplots(figsize=(fig_w, 0.35 * k + 2.5))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(-0.5 * k - 1.5, 1.8)
    ax.axis("off")

    y_axis = 0.0
    ax.plot([lo, hi], [y_axis, y_axis], color="black", lw=1.0)
    for tick in range(lo, hi + 1):
        ax.plot([tick, tick], [y_axis, y_axis + 0.18], color="black", lw=1.0)
        ax.text(tick, y_axis + 0.32, str(tick), ha="center", va="bottom", fontsize=9)

    half = (k + 1) // 2
    for idx, (rank, name_i) in enumerate(zip(sorted_ranks, sorted_names)):
        if idx < half:
            y = -0.5 * (idx + 1) - 0.3
            ax.plot([rank, rank], [y_axis, y], color="black", lw=0.8)
            ax.plot([rank, lo - 0.3], [y, y], color="black", lw=0.8)
            ax.text(lo - 0.4, y, f"{name_i}  ({rank:.2f})",
                    ha="right", va="center", fontsize=9)
        else:
            y = -0.5 * (k - idx) - 0.3
            ax.plot([rank, rank], [y_axis, y], color="black", lw=0.8)
            ax.plot([rank, hi + 0.3], [y, y], color="black", lw=0.8)
            ax.text(hi + 0.4, y, f"({rank:.2f})  {name_i}",
                    ha="left", va="center", fontsize=9)

    # CD bar (top-right)
    cd_y = y_axis + 1.0
    bar_left = lo
    bar_right = min(hi, lo + cd)
    ax.plot([bar_left, bar_right], [cd_y, cd_y], color="black", lw=2.0)
    ax.plot([bar_left, bar_left], [cd_y - 0.08, cd_y + 0.08], color="black", lw=2.0)
    ax.plot([bar_right, bar_right], [cd_y - 0.08, cd_y + 0.08], color="black", lw=2.0)
    ax.text((bar_left + bar_right) / 2, cd_y + 0.15,
            f"CD = {cd:.3f}", ha="center", fontsize=9)

    # Cliques: groups of adjacent algorithms not significantly different
    cliques: List[Tuple[int, int]] = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and (sorted_ranks[j + 1] - sorted_ranks[i]) <= cd + 1e-9:
            j += 1
        if j > i:
            cliques.append((i, j))
        i += 1
    maximal: List[Tuple[int, int]] = []
    for a, b in cliques:
        if not any(a2 <= a and b2 >= b and (a2, b2) != (a, b) for a2, b2 in cliques):
            maximal.append((a, b))

    base_y = -0.18
    step = 0.13
    for ci, (a, b) in enumerate(maximal):
        y = base_y - ci * step
        ax.plot(
            [sorted_ranks[a] - 0.05, sorted_ranks[b] + 0.05],
            [y, y],
            color="crimson",
            lw=3.0,
            solid_capstyle="butt",
        )

    sub = title
    if friedman_p_value is not None and not np.isnan(friedman_p_value):
        sub = f"{title}\nFriedman p = {friedman_p_value:.3g}"
    ax.set_title(sub, fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_dir, name)


CD_TARGETS: List[Tuple[str, str]] = [
    ("BinaryVector", "hamming_accuracy"),
    ("BinaryVector", "subset0_1"),
    ("ScoreVector", "auc_macro"),
    ("ScoreVector", "auprc_macro"),
]


def make_cd_diagrams(
    long_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    out_dir: str,
) -> Dict[str, int]:
    """Return mapping figure-id -> number of datasets used (for the report)."""
    dropped: Dict[str, int] = {}
    for ptype, metric in CD_TARGETS:
        for noise in NOISE_LEVELS:
            pivot = pivot_metric(long_df, ptype, metric, noise)
            if pivot.empty:
                print(f"[warn] no data for CD {ptype}/{metric}/{noise}")
                continue
            n_total = pivot.shape[0]
            # Keep algorithms present in at least half the datasets.
            valid = pivot.columns[pivot.notna().sum(axis=0) >= max(2, n_total // 2)]
            pivot = pivot[valid]
            complete = pivot.dropna(axis=0, how="any")
            n_datasets, k = complete.shape
            if n_datasets < 2 or k < 3:
                print(f"[warn] insufficient data for CD {ptype}/{metric}/{noise}: "
                      f"n={n_datasets}, k={k}")
                continue

            higher = metric_direction(metric) == 1
            ranks = rank_matrix(complete.values, higher_is_better=higher)
            avg_ranks = np.nanmean(ranks, axis=0)
            algs = [display_name(c) for c in complete.columns]
            cd = critical_difference(k, n_datasets)

            title = f"{ptype} / {metric} / noise {noise}  (N={n_datasets}, k={k})"
            name = f"cd_{ptype}_{metric}_noise{noise}"
            p = friedman_p(sig_df, ptype, metric, noise)
            plot_cd_diagram(avg_ranks, algs, cd, title, out_dir, name, p)
            dropped[name] = n_total - n_datasets
    return dropped


# ---------------------------------------------------------------------------
# 2) Noise-robustness curves
# ---------------------------------------------------------------------------


def make_noise_curves(results_dir: str, out_dir: str) -> None:
    targets: List[Tuple[str, str, str]] = [
        ("BinaryVector", "hamming_accuracy", "Hamming accuracy"),
        ("ScoreVector", "auc_macro", "AUC (macro)"),
    ]
    stats_dir = os.path.join(results_dir, "stats")
    for ptype, metric, pretty in targets:
        rows = []
        for noise in NOISE_LEVELS:
            path = os.path.join(stats_dir, f"avg_ranks_{ptype}_{metric}_{noise}.csv")
            if not os.path.exists(path):
                print(f"[warn] missing {path}")
                continue
            df = pd.read_csv(path)
            df["noise"] = float(noise)
            rows.append(df)
        if not rows:
            continue
        all_df = pd.concat(rows, ignore_index=True)
        all_df["display"] = all_df["algorithm"].map(display_name)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        bopos_palette = sns.color_palette("tab10", n_colors=len(IA_SHORT_NAMES))
        baseline_palette = sns.color_palette("Greys", n_colors=len(BASELINES_ORDER) + 2)[2:]

        # Plot BOPOs solid
        for i, (short, full) in enumerate(IA_ORDER):
            sub = all_df[all_df["algorithm"] == full].sort_values("noise")
            if sub.empty:
                continue
            ax.plot(sub["noise"], sub["avg_rank"], marker="o", lw=1.8,
                    color=bopos_palette[i], label=short)
        # Plot baselines dashed grayscale
        for i, b in enumerate(BASELINES_ORDER):
            sub = all_df[all_df["algorithm"] == b].sort_values("noise")
            if sub.empty:
                continue
            ax.plot(sub["noise"], sub["avg_rank"], marker="s", lw=1.6,
                    color=baseline_palette[i], linestyle="--",
                    label=BASELINE_DISPLAY[b])

        ax.set_xlabel("Label noise rate")
        ax.set_ylabel("Average rank across datasets (lower is better)")
        ax.set_title(f"Rank vs noise — {pretty}")
        ax.invert_yaxis()  # rank 1 (best) at top
        ax.set_xticks([0.0, 0.1, 0.2, 0.3])
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  frameon=False, fontsize=8)
        fig.tight_layout()
        save_figure(fig, out_dir, f"noise_curve_{metric}")


# ---------------------------------------------------------------------------
# 3) Abstention frontier
# ---------------------------------------------------------------------------


def make_abstention_plots(long_df: pd.DataFrame, out_dir: str) -> None:
    pa = long_df[long_df["ptype"] == "PartialAbstention"].copy()
    if pa.empty:
        print("[warn] no PartialAbstention data")
        return
    cols_present = set(pa["metric"].unique())

    if {"rec", "abs"}.issubset(cols_present):
        _scatter_two(pa, "abs", "rec",
                     xlabel="Abstention rate (abs, lower = better)",
                     ylabel="Recall (rec, higher = better)",
                     title="Abstention vs recall",
                     name="abstention_rec_vs_abs",
                     out_dir=out_dir)
    else:
        # Fallback per spec: use hamming_accuracy_pa instead
        if {"hamming_accuracy_pa", "abs"}.issubset(cols_present):
            _scatter_two(pa, "abs", "hamming_accuracy_pa",
                         xlabel="Abstention rate (abs, lower = better)",
                         ylabel="Hamming accuracy (pa)",
                         title="Abstention vs accuracy (recall column missing)",
                         name="abstention_rec_vs_abs",
                         out_dir=out_dir)

    if {"arec", "aabs"}.issubset(cols_present):
        _scatter_two(pa, "aabs", "arec",
                     xlabel="aabs (lower = better)",
                     ylabel="arec (higher = better)",
                     title="aabs vs arec frontier",
                     name="abstention_arec_vs_aabs",
                     out_dir=out_dir)


def _scatter_two(pa: pd.DataFrame, x_metric: str, y_metric: str,
                 xlabel: str, ylabel: str, title: str,
                 name: str, out_dir: str) -> None:
    xs = pa[pa["metric"] == x_metric]
    ys = pa[pa["metric"] == y_metric]
    merged = xs.merge(
        ys, on=["dataset", "ptype", "algorithm", "noise"],
        suffixes=("_x", "_y"),
    )
    merged = merged.dropna(subset=["mean_x", "mean_y"])
    if merged.empty:
        print(f"[warn] no abstention data for {name}")
        return

    markers = ["o", "s", "D", "^", "v", "<", ">", "P"]
    alg_to_marker = {full: markers[i % len(markers)] for i, (_, full) in enumerate(IA_ORDER)}
    noise_palette = dict(zip(NOISE_LEVELS, sns.color_palette("viridis", n_colors=4)))

    fig, ax = plt.subplots(figsize=(8, 6))
    seen_alg = set()
    seen_noise = set()
    for _, r in merged.iterrows():
        alg = r["algorithm"]
        noise = str(r["noise"])
        color = noise_palette.get(noise, "gray")
        marker = alg_to_marker.get(alg, "o")
        ax.scatter(r["mean_x"], r["mean_y"], s=55, marker=marker,
                   color=color, edgecolor="black", lw=0.4, alpha=0.85)
        seen_alg.add(alg)
        seen_noise.add(noise)

    # Legends: one for algorithm (marker), one for noise (color)
    alg_handles = [
        Line2D([0], [0], marker=alg_to_marker[full], linestyle="None",
               markerfacecolor="white", markeredgecolor="black",
               markersize=9, label=display_name(full))
        for _, full in IA_ORDER if full in seen_alg
    ]
    noise_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=noise_palette[n], markeredgecolor="black",
               markersize=9, label=f"noise={n}")
        for n in NOISE_LEVELS if n in seen_noise
    ]
    leg1 = ax.legend(handles=alg_handles, title="Algorithm",
                     loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     fontsize=8, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=noise_handles, title="Noise",
              loc="upper left", bbox_to_anchor=(1.01, 0.45),
              fontsize=8, frameon=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    save_figure(fig, out_dir, name)


# ---------------------------------------------------------------------------
# 4) Hamming vs Subset trade-off
# ---------------------------------------------------------------------------


def make_tradeoff_plot(long_df: pd.DataFrame, out_dir: str) -> None:
    bv = long_df[long_df["ptype"] == "BinaryVector"]
    hh = bv[bv["metric"] == "hamming_accuracy"]
    ss = bv[bv["metric"] == "subset0_1"]
    if hh.empty or ss.empty:
        print("[warn] tradeoff: missing metrics")
        return
    h_mean = hh.groupby(["algorithm", "noise"])["mean"].mean().reset_index(name="hamming")
    s_mean = ss.groupby(["algorithm", "noise"])["mean"].mean().reset_index(name="subset")
    merged = h_mean.merge(s_mean, on=["algorithm", "noise"])

    bopos_palette = sns.color_palette("tab10", n_colors=len(IA_SHORT_NAMES))
    baseline_palette = sns.color_palette("Set2", n_colors=len(BASELINES_ORDER))
    color_map: Dict[str, tuple] = {}
    marker_map: Dict[str, str] = {}
    for i, (short, full) in enumerate(IA_ORDER):
        color_map[full] = bopos_palette[i]
        marker_map[full] = "o"
    for i, b in enumerate(BASELINES_ORDER):
        color_map[b] = baseline_palette[i]
        marker_map[b] = "s"

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for alg, group in merged.groupby("algorithm"):
        group = group.copy()
        group["noise_f"] = group["noise"].astype(float)
        group = group.sort_values("noise_f")
        if alg not in color_map:
            continue
        xs = group["hamming"].values
        ys = group["subset"].values
        c = color_map[alg]
        m = marker_map[alg]
        ls = "-" if is_bopos(alg) else "--"
        ax.plot(xs, ys, marker=m, color=c, lw=1.2, ls=ls,
                markersize=6, label=display_name(alg), alpha=0.9)
        # Arrows from noise 0.0 -> 0.3 (consecutive segments)
        for i in range(len(xs) - 1):
            ax.annotate("",
                        xy=(xs[i + 1], ys[i + 1]),
                        xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=0.9, alpha=0.6))

    ax.set_xlabel("Mean Hamming accuracy (higher = better)")
    ax.set_ylabel("Mean Subset accuracy (subset0_1, higher = better)")
    ax.set_title("Hamming vs Subset trade-off (arrows: noise 0.0 → 0.3)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir, "hamming_vs_subset_tradeoff")


# ---------------------------------------------------------------------------
# 5) Heatmaps
# ---------------------------------------------------------------------------


HEATMAP_METRICS: List[Tuple[str, str]] = [
    ("BinaryVector", "hamming_accuracy"),
    ("BinaryVector", "subset0_1"),
    ("ScoreVector", "auc_macro"),
    ("ScoreVector", "auprc_macro"),
]


def make_avg_rank_heatmap(long_df: pd.DataFrame, out_dir: str) -> None:
    columns: List[str] = []
    rank_cols: Dict[str, pd.Series] = {}
    for ptype, metric in HEATMAP_METRICS:
        for noise in NOISE_LEVELS:
            pivot = pivot_metric(long_df, ptype, metric, noise)
            if pivot.empty:
                continue
            n_total = pivot.shape[0]
            valid = pivot.columns[pivot.notna().sum(axis=0) >= max(2, n_total // 2)]
            pivot = pivot[valid]
            complete = pivot.dropna(axis=0, how="any")
            if complete.shape[0] < 2 or complete.shape[1] < 3:
                continue
            higher = metric_direction(metric) == 1
            ranks = rank_matrix(complete.values, higher_is_better=higher)
            avg_ranks = pd.Series(np.nanmean(ranks, axis=0), index=complete.columns)
            col_name = f"{metric}\n@{noise}"
            columns.append(col_name)
            rank_cols[col_name] = avg_ranks
    if not rank_cols:
        print("[warn] avg_rank_heatmap: no data")
        return

    # Build full row set (BOPOs + baselines)
    row_keys: List[str] = [full for _, full in IA_ORDER] + BASELINES_ORDER
    row_labels: List[str] = [display_name(k) for k in row_keys]

    mat = np.full((len(row_keys), len(columns)), np.nan)
    for j, col in enumerate(columns):
        s = rank_cols[col]
        for i, alg in enumerate(row_keys):
            if alg in s.index:
                mat[i, j] = s[alg]

    fig_w = max(10, 0.8 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    sns.heatmap(
        mat,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        xticklabels=columns,
        yticklabels=row_labels,
        cbar_kws={"label": "Average rank (1 = best)"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Algorithm")
    ax.set_title("Average rank heatmap — algorithms vs (metric, noise)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_figure(fig, out_dir, "avg_rank_heatmap")


def make_best_variant_heatmap(long_df: pd.DataFrame, out_dir: str) -> None:
    bv = long_df[(long_df["ptype"] == "BinaryVector")
                 & (long_df["metric"] == "hamming_accuracy")]
    if bv.empty:
        print("[warn] best_variant_per_dataset: no data")
        return
    # Restrict to BOPOs variants
    bopos = bv[bv["algorithm"].isin(IA_TO_ALG.values())]
    if bopos.empty:
        return
    # For each (dataset, noise) pick the BOPOs algo with the best mean.
    grouped = bopos.groupby(["dataset", "noise"])
    best_rows = []
    for (ds, noise), g in grouped:
        g = g.dropna(subset=["mean"])
        if g.empty:
            continue
        best = g.loc[g["mean"].idxmax()]
        best_rows.append({
            "dataset": ds,
            "noise": noise,
            "best_full": best["algorithm"],
            "best_ia": ALG_TO_IA.get(best["algorithm"], "?"),
        })
    if not best_rows:
        print("[warn] best_variant_per_dataset: nothing to plot")
        return
    bdf = pd.DataFrame(best_rows)
    ia_idx = {ia: i for i, ia in enumerate(IA_SHORT_NAMES)}

    datasets = sorted(bdf["dataset"].unique())
    mat = np.full((len(datasets), len(NOISE_LEVELS)), np.nan)
    labels = np.full((len(datasets), len(NOISE_LEVELS)), "", dtype=object)
    for i, ds in enumerate(datasets):
        for j, noise in enumerate(NOISE_LEVELS):
            row = bdf[(bdf["dataset"] == ds) & (bdf["noise"] == noise)]
            if row.empty:
                continue
            ia = row.iloc[0]["best_ia"]
            mat[i, j] = ia_idx.get(ia, np.nan)
            labels[i, j] = ia

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(datasets) + 1.5))
    cmap = sns.color_palette("tab10", n_colors=len(IA_SHORT_NAMES))
    cmap_lut = matplotlib.colors.ListedColormap(cmap)
    sns.heatmap(
        mat,
        ax=ax,
        annot=labels,
        fmt="",
        cmap=cmap_lut,
        vmin=-0.5,
        vmax=len(IA_SHORT_NAMES) - 0.5,
        xticklabels=NOISE_LEVELS,
        yticklabels=datasets,
        cbar_kws={"label": "Best BOPOs variant"},
        linewidths=0.5,
        linecolor="white",
    )
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.arange(len(IA_SHORT_NAMES)))
    cbar.set_ticklabels(IA_SHORT_NAMES)
    ax.set_xlabel("Noise rate")
    ax.set_ylabel("Dataset")
    ax.set_title("Best BOPOs variant per dataset × noise (hamming_accuracy)")
    fig.tight_layout()
    save_figure(fig, out_dir, "best_variant_per_dataset")


# ---------------------------------------------------------------------------
# FIGURES.md writer
# ---------------------------------------------------------------------------


FIGURES_MD_TEMPLATE = """# v2 Result Figures

Generated by `utils/plot_figures.py`. Regenerate with:

```bash
python utils/plot_figures.py \\
    --results_dir results/final_20260514_v2_summary \\
    --raw_results_dir results/20260514_v2
```

- **Datasets (9):** chd_49, emotions, scene, viruspseaac, yeast, water-quality,
  humanpseaac, gpositivepseaac, plantpseaac.
- **Noise levels:** 0.0, 0.1, 0.2, 0.3.
- **Repeats × folds:** 5 × 5.
- **Algorithms:** 8 BOPOs variants (IA1–IA8) + 6 baselines (BR, CC, CLR, MLkNN,
  ECC, LP).
- **Metric directions:** taken from `utils/statistical_tests.py`
  (`HIGHER_IS_BETTER` / `LOWER_IS_BETTER`).

## IA Variant Legend
| Short | Full algorithm                       |
| ----- | ------------------------------------ |
| IA1   | PreOrder × Hamming × height=None     |
| IA2   | PreOrder × Hamming × height=2        |
| IA3   | PreOrder × Subset × height=None      |
| IA4   | PreOrder × Subset × height=2         |
| IA5   | PartialOrder × Hamming × height=None |
| IA6   | PartialOrder × Hamming × height=2    |
| IA7   | PartialOrder × Subset × height=None  |
| IA8   | PartialOrder × Subset × height=2     |

(Ordering matches `evaluation_test.py::EvaluationConfig.INFERENCE_ALGORITHMS`.)

## Figures

### 1. CD Diagrams (`cd_diagrams/`)
Files: `cd_<ptype>_<metric>_noise<n>.{{pdf,tex,png}}` — 16 total
(BinaryVector × {{hamming_accuracy, subset0_1}}, ScoreVector × {{auc_macro,
auprc_macro}}, each at 4 noise levels).

- **Source data:** per-dataset rank matrix built from `*_summary.csv`.
- **Axes:** rank, smaller is better; ranks are computed per-dataset with
  `scipy.stats.rankdata(method="average")` after sign-flipping so that rank 1
  is the best.
- **Bars:** red horizontal bars connect cliques of algorithms whose pairwise
  rank gap is ≤ CD (Nemenyi, α = 0.05).
- **Annotations:** Friedman p-value pulled from
  `stats/significance_summary.csv`.
- **LaTeX include:**
  `\\input{{figures/cd_diagrams/cd_BinaryVector_hamming_accuracy_noise0.0.tex}}`
- **Takeaway:** <fill in>

### 2. Noise-robustness curves (`noise_curves/`)
Files: `noise_curve_hamming_accuracy.{{pdf,tex,png}}`,
`noise_curve_auc_macro.{{pdf,tex,png}}` — 2 total.

- **Source data:** `stats/avg_ranks_<ptype>_<metric>_<noise>.csv`.
- **Axes:** x = noise rate ∈ {{0.0, 0.1, 0.2, 0.3}}, y = avg rank
  (y-axis inverted so best ranks are on top).
- **Lines:** 8 BOPOs variants drawn solid (`tab10` palette), 6 baselines dashed
  (greyscale).
- **Takeaway:** <fill in>

### 3. Abstention frontier (`abstention/`)
Files: `abstention_rec_vs_abs.{{pdf,tex,png}}`,
`abstention_arec_vs_aabs.{{pdf,tex,png}}` — 2 total.

- **Source data:** `*_PartialAbstention_summary.csv` (only the 8 BOPOs
  variants — baselines cannot abstain).
- **Axes:** abstention rate vs recall-style metric; one scatter point per
  (algorithm, dataset, noise).
- **Encoding:** marker shape = algorithm (IA1–IA8), color = noise level
  (viridis).
- **Takeaway:** <fill in>

### 4. Hamming vs Subset trade-off (`tradeoff/`)
File: `hamming_vs_subset_tradeoff.{{pdf,tex,png}}` — 1 total.

- **Source data:** BinaryVector summaries.
- **Axes:** x = mean Hamming accuracy across datasets, y = mean Subset
  accuracy (`subset0_1`, exact match — higher is better, per
  `statistical_tests.py`).
- **Trajectories:** for each algorithm, 4 points (one per noise level)
  connected by arrows showing the noise 0.0 → 0.3 path. BOPOs solid, baselines
  dashed.
- **Takeaway:** <fill in>

### 5. Heatmaps (`heatmaps/`)
Files: `avg_rank_heatmap.{{pdf,tex,png}}`,
`best_variant_per_dataset.{{pdf,tex,png}}` — 2 total.

- **`avg_rank_heatmap`:** rows = 14 algorithms, columns = 16 (metric × noise)
  cells, color = average rank (green = best). Useful single-glance summary of
  how each algorithm fares across metrics and noise levels.
- **`best_variant_per_dataset`:** rows = 9 datasets, columns = 4 noise
  levels, cell label/color = the BOPOs IA variant with the highest
  `hamming_accuracy` for that combination. Diagnostic for whether a single
  variant dominates or whether the best variant is dataset-dependent.
- **Takeaway:** <fill in>

## Notes on regeneration

- TikZ exports use `matplot2tikz.save(...)`. Some plots (notably seaborn
  heatmaps and the CD diagrams with custom shapes) may export imperfectly;
  the script catches failures, prints a warning, and continues with
  PDF + PNG.
- Dropped datasets per CD diagram (due to NaN cells) are logged at run time
  and summarised in the script's final stdout report.
"""


def write_figures_md(out_dir: str) -> None:
    path = os.path.join(out_dir, "FIGURES.md")
    with open(path, "w") as f:
        f.write(FIGURES_MD_TEMPLATE)
    print(f"[ok] wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results/final_20260514_v2_summary")
    parser.add_argument("--raw_results_dir", default="results/20260514_v2",
                        help="(reserved) location of the raw pickle results")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    figures_root = os.path.join(results_dir, "figures")
    os.makedirs(figures_root, exist_ok=True)

    print(f"[info] loading summaries from {results_dir}")
    long_df = load_all_summaries(results_dir)
    sig_df = load_significance_summary(results_dir)
    print(f"[info] long DataFrame: {len(long_df):,} rows; "
          f"significance summary: {len(sig_df):,} rows")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # 1) CD diagrams
        cd_dir = os.path.join(figures_root, "cd_diagrams")
        dropped = make_cd_diagrams(long_df, sig_df, cd_dir)

        # 2) Noise curves
        nc_dir = os.path.join(figures_root, "noise_curves")
        make_noise_curves(results_dir, nc_dir)

        # 3) Abstention
        ab_dir = os.path.join(figures_root, "abstention")
        make_abstention_plots(long_df, ab_dir)

        # 4) Tradeoff
        tr_dir = os.path.join(figures_root, "tradeoff")
        make_tradeoff_plot(long_df, tr_dir)

        # 5) Heatmaps
        hm_dir = os.path.join(figures_root, "heatmaps")
        make_avg_rank_heatmap(long_df, hm_dir)
        make_best_variant_heatmap(long_df, hm_dir)

    write_figures_md(figures_root)

    # Final tally
    counts: Dict[str, int] = {}
    for sub in ("cd_diagrams", "noise_curves", "abstention", "tradeoff", "heatmaps"):
        d = os.path.join(figures_root, sub)
        if os.path.isdir(d):
            counts[sub] = len([f for f in os.listdir(d) if f.endswith(".pdf")])
        else:
            counts[sub] = 0
    print("\n=== Figure counts (PDF) ===")
    for sub, n in counts.items():
        print(f"  {sub:14s} {n}")
    if dropped:
        any_drop = {k: v for k, v in dropped.items() if v > 0}
        if any_drop:
            print("\nDatasets dropped from CD diagrams (NaN rows):")
            for k, v in any_drop.items():
                print(f"  {k}: {v} datasets dropped")
        else:
            print("\nNo datasets dropped from CD diagrams.")


if __name__ == "__main__":
    main()
