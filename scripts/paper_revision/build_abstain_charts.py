#!/usr/bin/env python3
"""Build abstention-benefit charts for paper_revision/.

Two chart families, both supporting the claim that abstaining on uncertain
labels yields better quality on retained labels than standard MLC:

  A. Coverage-risk curve. x=abstention rate (PA metric ``abs``),
     y=quality on retained labels (default ``f1_pa``, can swap to
     ``jaccard_pa`` once the next eval pass populates it). Each PA/PR
     method = one dot per noise level. Baselines (BR/CC/CLR/ECC) plotted
     as horizontal reference lines at their BV quality (since they cannot
     abstain).

  B. Paired bars. Per method, two bars side-by-side at noise=0.0:
     - left: BV quality (default ``f1``)
     - right: PA quality (``f1_pa``) — only present for the 8 order-based
       methods that can abstain
     Gap = empirical benefit of abstention.

Layout::

    paper_revision/figures_<style>/<learner>/abstain/
        aggregate/{coverage_risk,paired_bars}.pdf
        per_dataset/<ds>/{coverage_risk,paired_bars}.pdf

Usage::

    python scripts/paper_revision/build_abstain_charts.py --style enhanced \
        --output_dir paper_revision/figures_enhanced
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_panels import (  # noqa: E402
    DATASETS,
    LEARNERS,
    METHOD_ORDER,
    NOISE_COLORS_ENHANCED,
    NOISE_COLORS_ORIGINAL,
    NOISE_LEVELS,
    load_long_df,
)

PA_METHODS = METHOD_ORDER[:8]
BASELINE_METHODS = METHOD_ORDER[8:]


def _aggregate_value(df: pd.DataFrame, algo: str, metric: str, noise: str) -> float:
    sel = df[(df.algorithm == algo) & (df.metric == metric) & (df.noise == noise)]
    vals = sel.value.dropna().to_numpy()
    return float(np.mean(vals)) if vals.size else float("nan")


def _baseline_for(pa_metric: str, bv_metric: str) -> tuple[str, str]:
    """Return (effective_bv_metric, baseline_label) for fair comparison.

    For jaccard_pa we route the baseline to the saved ``jaccard`` BV column
    (computed per-instance and averaged by ``evaluate.py``), rather than
    deriving J from F1 at the aggregate level --- the F1->J identity
    ``J = F1/(2-F1)`` holds per-instance but Jensen's inequality means
    ``mean(J_i) >= mean(F1_i)/(2-mean(F1_i))``, so deriving would
    under-estimate Jaccard whenever per-instance F1 has non-zero variance.
    """
    if pa_metric == "jaccard_pa" and bv_metric == "f1":
        return ("jaccard", "jaccard")
    return (bv_metric, bv_metric)


def _bv_value(df: pd.DataFrame, algo: str, pa_metric: str, bv_metric: str,
              noise: str) -> float:
    """Aggregate baseline value using the per-pa_metric effective BV column."""
    eff_metric, _ = _baseline_for(pa_metric, bv_metric)
    return _aggregate_value(df, algo, eff_metric, noise)


def render_coverage_risk(
    df: pd.DataFrame,
    pa_metric: str,
    bv_metric: str,
    out_pdf: Path,
    title: str,
    style: str,
) -> None:
    """Render coverage-risk scatter for one (learner, scope) slice.

    df: long-format rows already filtered to (base_learner, optional dataset).
    """
    palette = NOISE_COLORS_ENHANCED if style == "enhanced" else NOISE_COLORS_ORIGINAL
    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    # Track baselines (no abstention) — plot as horizontal reference lines.
    eff_bv, bv_label = _baseline_for(pa_metric, bv_metric)
    baseline_qualities: dict[str, float] = {}
    for algo, _label in BASELINE_METHODS:
        # average across noise levels for the reference line
        vals = df[(df.algorithm == algo) & (df.metric == eff_bv)].value.dropna()
        if vals.size:
            baseline_qualities[algo] = float(vals.mean()) * 100.0

    best_baseline = max(baseline_qualities.values(), default=float("nan"))
    if np.isfinite(best_baseline):
        ax.axhline(
            y=best_baseline,
            color="grey",
            linestyle="--",
            linewidth=0.7,
            alpha=0.8,
            label=f"best standard-MLC {bv_label} (no abstention)",
        )

    # PA/PR methods as scatter, colored by noise.
    marker_per_method = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for m_idx, (algo, label) in enumerate(PA_METHODS):
        for noise in NOISE_LEVELS:
            abs_v = _aggregate_value(df, algo, "abs", noise)
            qual_v = _aggregate_value(df, algo, pa_metric, noise)
            if not (np.isfinite(abs_v) and np.isfinite(qual_v)):
                continue
            ax.scatter(
                abs_v * 100.0,
                qual_v * 100.0,
                color=palette[noise],
                marker=marker_per_method[m_idx],
                s=22,
                edgecolors="black",
                linewidths=0.3,
            )

    # Custom legend: one marker per method, one color row per noise.
    method_handles = [
        plt.Line2D(
            [],
            [],
            marker=marker_per_method[i],
            linestyle="None",
            color="dimgrey",
            markersize=5,
            label=PA_METHODS[i][1],
        )
        for i in range(len(PA_METHODS))
    ]
    noise_handles = [
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="None",
            color=palette[n],
            markersize=6,
            label=fr"$\alpha={n}$",
        )
        for n in NOISE_LEVELS
    ]
    leg1 = ax.legend(
        handles=method_handles,
        fontsize=5,
        loc="lower right",
        ncol=2,
        framealpha=0.85,
        handletextpad=0.3,
        columnspacing=0.6,
        borderpad=0.3,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=noise_handles,
        fontsize=5,
        loc="upper left",
        framealpha=0.85,
        handletextpad=0.3,
        borderpad=0.3,
    )

    ax.set_xlabel(
        "Abstention rate (% labels skipped, higher = more conservative)",
        fontsize=6,
    )
    ax.set_ylabel(
        f"{pa_metric} on predicted labels (%, higher = better)", fontsize=6,
    )
    ax.set_title(title, fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_paired_bars(
    df: pd.DataFrame,
    pa_metric: str,
    bv_metric: str,
    out_pdf: Path,
    title: str,
    style: str,
) -> None:
    """Render BV-vs-PA paired bars for all 12 methods, 2x2 grid over noise."""
    palette = NOISE_COLORS_ENHANCED if style == "enhanced" else NOISE_COLORS_ORIGINAL
    bv_color = "#9ecae1"  # light blue for standard MLC
    n_methods = len(METHOD_ORDER)

    _, bv_label = _baseline_for(pa_metric, bv_metric)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 4.6), sharey=True)
    for ax, noise in zip(axes.flat, NOISE_LEVELS):
        bv_vals = np.full(n_methods, np.nan)
        pa_vals = np.full(n_methods, np.nan)
        for i, (algo, _) in enumerate(METHOD_ORDER):
            bv_vals[i] = _bv_value(df, algo, pa_metric, bv_metric, noise) * 100.0
            if i < 8:
                pa_vals[i] = _aggregate_value(df, algo, pa_metric, noise) * 100.0

        x = np.arange(n_methods)
        width = 0.4
        ax.bar(
            x - width / 2, bv_vals, width,
            color=bv_color,
            label=f"Standard MLC ({bv_label}, no abstention)",
        )
        ax.bar(
            x + width / 2, pa_vals, width,
            color=palette[noise],
            label=f"With abstention ({pa_metric} on retained labels)",
        )
        for i in range(8):
            if np.isfinite(bv_vals[i]) and np.isfinite(pa_vals[i]):
                gap = pa_vals[i] - bv_vals[i]
                if abs(gap) >= 0.05:
                    color = "darkgreen" if gap > 0 else "firebrick"
                    sign = "+" if gap > 0 else ""
                    ax.annotate(
                        f"{sign}{gap:.1f}",
                        xy=(x[i] + width / 2, pa_vals[i]),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        fontsize=5,
                        color=color,
                    )
        ax.axvline(x=7.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m[1] for m in METHOD_ORDER], rotation=-35, ha="left", fontsize=5,
        )
        ax.set_title(fr"$\alpha={noise}$", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.legend(fontsize=5, loc="lower left", framealpha=0.85, borderpad=0.3)
        ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0, 0].set_ylabel("score (%)", fontsize=7)
    axes[1, 0].set_ylabel("score (%)", fontsize=7)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0, 0.96))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _pareto_front(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return boolean mask of points on the Pareto front (max y, min x).

    A point (xi, yi) is dominated if some (xj, yj) satisfies
    xj <= xi and yj >= yi with strict inequality somewhere.
    """
    n = xs.size
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(xs[i]) or not np.isfinite(ys[i]):
            on_front[i] = False
            continue
        for j in range(n):
            if i == j or not (np.isfinite(xs[j]) and np.isfinite(ys[j])):
                continue
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                on_front[i] = False
                break
    return on_front


def render_coverage_risk_pareto(
    df: pd.DataFrame,
    pa_metric: str,
    bv_metric: str,
    out_pdf: Path,
    title: str,
    style: str,
) -> None:
    """Coverage-risk with Pareto frontier highlighted + win-region shaded.

    Win region: y > best_baseline_quality. Pareto points connected with a
    grey line; dominated points faded.
    """
    palette = NOISE_COLORS_ENHANCED if style == "enhanced" else NOISE_COLORS_ORIGINAL
    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    eff_bv, bv_label = _baseline_for(pa_metric, bv_metric)
    baseline_qualities: list[float] = []
    for algo, _ in BASELINE_METHODS:
        vals = df[(df.algorithm == algo) & (df.metric == eff_bv)].value.dropna()
        if vals.size:
            baseline_qualities.append(float(vals.mean()) * 100.0)
    best_baseline = max(baseline_qualities, default=float("nan"))

    xs_all: list[float] = []
    ys_all: list[float] = []
    colors_all: list[str] = []
    for algo, _ in PA_METHODS:
        for noise in NOISE_LEVELS:
            a = _aggregate_value(df, algo, "abs", noise)
            q = _aggregate_value(df, algo, pa_metric, noise)
            if np.isfinite(a) and np.isfinite(q):
                xs_all.append(a * 100.0)
                ys_all.append(q * 100.0)
                colors_all.append(palette[noise])
    xs = np.array(xs_all)
    ys = np.array(ys_all)

    # Shade win region.
    if np.isfinite(best_baseline):
        ax.axhspan(
            best_baseline, max(ys.max() if ys.size else best_baseline, best_baseline) + 5,
            color="#c7e9c0", alpha=0.4, zorder=0,
        )
        ax.axhline(
            y=best_baseline, color="grey", linestyle="--", linewidth=0.7,
            label=f"best standard-MLC {bv_label}",
        )

    if xs.size:
        mask = _pareto_front(xs, ys)
        # Dominated points: faded.
        for i in range(xs.size):
            if not mask[i]:
                ax.scatter(xs[i], ys[i], color=colors_all[i], s=16, alpha=0.25,
                           edgecolors="none")
        # Frontier: solid + connecting line.
        idx = np.where(mask)[0]
        order = sorted(idx, key=lambda k: xs[k])
        if len(order) >= 2:
            ax.plot([xs[k] for k in order], [ys[k] for k in order],
                    color="black", linewidth=0.7, alpha=0.6, zorder=2)
        for k in order:
            ax.scatter(xs[k], ys[k], color=colors_all[k], s=28,
                       edgecolors="black", linewidths=0.4, zorder=3)

    ax.set_xlabel("Abstention rate (%)", fontsize=6)
    ax.set_ylabel(f"{pa_metric} on retained (%)", fontsize=6)
    ax.set_title(f"{title}  -- Pareto frontier + win-region", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    ax.legend(fontsize=5, loc="lower right", framealpha=0.85, borderpad=0.3)
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_gain_heatmap(
    df: pd.DataFrame,
    pa_metric: str,
    bv_metric: str,
    out_pdf: Path,
    title: str,
    style: str,
) -> None:
    """8x4 heatmap: gain = pa_metric - bv_metric, per (method, noise)."""
    _, bv_label = _baseline_for(pa_metric, bv_metric)
    gain = np.full((len(PA_METHODS), len(NOISE_LEVELS)), np.nan)
    for i, (algo, _) in enumerate(PA_METHODS):
        for j, noise in enumerate(NOISE_LEVELS):
            pa = _aggregate_value(df, algo, pa_metric, noise)
            bv = _bv_value(df, algo, pa_metric, bv_metric, noise)
            if np.isfinite(pa) and np.isfinite(bv):
                gain[i, j] = (pa - bv) * 100.0

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    vmax = max(np.nanmax(np.abs(gain)), 1.0) if np.any(np.isfinite(gain)) else 1.0
    im = ax.imshow(gain, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(PA_METHODS)):
        for j in range(len(NOISE_LEVELS)):
            if np.isfinite(gain[i, j]):
                ax.text(
                    j, i, f"{gain[i, j]:+.1f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if abs(gain[i, j]) > 0.6 * vmax else "black",
                )
    ax.set_xticks(np.arange(len(NOISE_LEVELS)))
    ax.set_xticklabels([fr"$\alpha={n}$" for n in NOISE_LEVELS], fontsize=6)
    ax.set_yticks(np.arange(len(PA_METHODS)))
    ax.set_yticklabels([m[1] for m in PA_METHODS], fontsize=6)
    ax.set_title(f"{title}  -- gain ({pa_metric} - {bv_label}, percentage points)",
                 fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=5)
    fig.tight_layout(pad=0.2)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_delta_bars(
    df: pd.DataFrame,
    pa_metric: str,
    bv_metric: str,
    out_pdf: Path,
    title: str,
    style: str,
) -> None:
    """2x2 grid of gain bars (one bar per PA method, signed)."""
    _, bv_label = _baseline_for(pa_metric, bv_metric)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 4.4), sharey=True)
    for ax, noise in zip(axes.flat, NOISE_LEVELS):
        gains = np.full(len(PA_METHODS), np.nan)
        for i, (algo, _) in enumerate(PA_METHODS):
            pa = _aggregate_value(df, algo, pa_metric, noise)
            bv = _bv_value(df, algo, pa_metric, bv_metric, noise)
            if np.isfinite(pa) and np.isfinite(bv):
                gains[i] = (pa - bv) * 100.0
        colors = ["#1a9850" if g >= 0 else "#d73027" for g in gains]
        x = np.arange(len(PA_METHODS))
        ax.bar(x, gains, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)
        for i, g in enumerate(gains):
            if np.isfinite(g):
                ax.text(
                    i, g + (0.3 if g >= 0 else -0.6),
                    f"{g:+.1f}", ha="center", fontsize=5,
                    color="darkgreen" if g >= 0 else "darkred",
                )
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in PA_METHODS], rotation=-35, ha="left",
                           fontsize=5)
        ax.set_title(fr"$\alpha={noise}$", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0, 0].set_ylabel(f"gain (pp): {pa_metric} - {bv_label}", fontsize=6)
    axes[1, 0].set_ylabel(f"gain (pp): {pa_metric} - {bv_label}", fontsize=6)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0, 0.96))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_efficiency_quality(
    df: pd.DataFrame,
    pa_metric: str,
    bv_metric: str,
    out_pdf: Path,
    title: str,
    style: str,
) -> None:
    """Scatter: x = coverage (1 - abs), y = pa_metric. Upper-right = robust."""
    palette = NOISE_COLORS_ENHANCED if style == "enhanced" else NOISE_COLORS_ORIGINAL
    _, bv_label = _baseline_for(pa_metric, bv_metric)
    fig, ax = plt.subplots(figsize=(3.8, 2.8))

    eff_bv, _ = _baseline_for(pa_metric, bv_metric)
    baseline_qualities = []
    for a, _ in BASELINE_METHODS:
        s = df[(df.algorithm == a) & (df.metric == eff_bv)].value.dropna()
        if s.size:
            baseline_qualities.append(float(s.mean()) * 100.0)
    best_baseline = max(baseline_qualities, default=float("nan"))
    if np.isfinite(best_baseline):
        ax.axhline(
            y=best_baseline, color="grey", linestyle="--", linewidth=0.7,
            label=f"best standard-MLC {bv_label}",
        )

    marker_per_method = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, (algo, _label) in enumerate(PA_METHODS):
        for noise in NOISE_LEVELS:
            a = _aggregate_value(df, algo, "abs", noise)
            q = _aggregate_value(df, algo, pa_metric, noise)
            if not (np.isfinite(a) and np.isfinite(q)):
                continue
            coverage = (1.0 - a) * 100.0
            ax.scatter(
                coverage, q * 100.0,
                color=palette[noise], marker=marker_per_method[i],
                s=24, edgecolors="black", linewidths=0.3,
            )
    ax.set_xlabel("Coverage = 1 - abstention (% labels predicted)", fontsize=6)
    ax.set_ylabel(f"{pa_metric} on retained (%)", fontsize=6)
    ax.set_title(f"{title}  -- robustness: upper-right = robust", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    ax.legend(fontsize=5, loc="lower left", framealpha=0.85, borderpad=0.3)
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_effective_f1(
    df: pd.DataFrame,
    pa_metric: str,
    bv_metric: str,
    out_pdf: Path,
    title: str,
    style: str,
) -> None:
    """Effective metric = (1 - abs) * pa_metric. Plot vs alpha for each method.

    Compares against baseline bv_metric line (per noise) so the reader can
    see when abstention's net effect (after penalising for skipped labels)
    still beats standard MLC.
    """
    _, bv_label = _baseline_for(pa_metric, bv_metric)
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    xs = np.arange(len(NOISE_LEVELS))

    # PA/PR methods.
    cmap = plt.get_cmap("tab10")
    for i, (algo, label) in enumerate(PA_METHODS):
        ys = []
        for noise in NOISE_LEVELS:
            a = _aggregate_value(df, algo, "abs", noise)
            q = _aggregate_value(df, algo, pa_metric, noise)
            ys.append((1 - a) * q * 100.0 if (np.isfinite(a) and np.isfinite(q)) else np.nan)
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.0,
                color=cmap(i % 10), label=label)

    # Best baseline reference (BV metric, per noise; converted if needed).
    best_baseline_per_noise = []
    for noise in NOISE_LEVELS:
        vals = [
            _bv_value(df, a, pa_metric, bv_metric, noise) * 100.0
            for a, _ in BASELINE_METHODS
        ]
        vals = [v for v in vals if np.isfinite(v)]
        best_baseline_per_noise.append(max(vals) if vals else float("nan"))
    ax.plot(xs, best_baseline_per_noise, color="grey", linestyle="--",
            linewidth=1.0, marker="s", markersize=3,
            label=f"best standard-MLC {bv_label}")

    ax.set_xticks(xs)
    ax.set_xticklabels([fr"$\alpha={n}$" for n in NOISE_LEVELS], fontsize=6)
    ax.set_ylabel(f"effective {pa_metric} = (1-abs) * {pa_metric} (%)", fontsize=6)
    ax.set_title(f"{title}  -- penalised quality (abstention costs labels)",
                 fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=4, loc="lower left", ncol=3, framealpha=0.85,
              borderpad=0.3, handletextpad=0.3, columnspacing=0.6)
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_abstention_rate(
    df: pd.DataFrame,
    pa_metric: str,
    bv_metric: str,
    out_pdf: Path,
    title: str,
    style: str,
) -> None:
    """2x2 grid: bar = `abs` (instance-level), dot overlay = `aabs` (label-level).

    Definitions match preorder4mlc/evaluation_metric.py:
        abs  = # instances with at least one abstained label / T
               (instance-level "any-abstention" rate, normally larger)
        aabs = total -1 cells / (T * K)
               (label-level / per-cell average abstention rate)

    Only the 8 PA/PR methods (baselines never abstain). Shares the noise-
    colour palette and method ordering with render_paired_bars so the two
    figures are reader-comparable side-by-side. Abstention rates depend
    only on the PA decision rule, not on which quality metric we measure,
    so this chart is metric-independent.
    """
    del pa_metric, bv_metric  # unused — abstention rates are metric-independent
    palette = NOISE_COLORS_ENHANCED if style == "enhanced" else NOISE_COLORS_ORIGINAL
    n_methods = len(PA_METHODS)

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 4.6), sharey=True)
    for ax, noise in zip(axes.flat, NOISE_LEVELS):
        abs_vals = np.full(n_methods, np.nan)
        aabs_vals = np.full(n_methods, np.nan)
        for i, (algo, _) in enumerate(PA_METHODS):
            abs_vals[i] = _aggregate_value(df, algo, "abs", noise) * 100.0
            aabs_vals[i] = _aggregate_value(df, algo, "aabs", noise) * 100.0

        x = np.arange(n_methods)
        ax.bar(
            x, abs_vals, width=0.6,
            color=palette[noise], edgecolor="black", linewidth=0.3,
            label="abs (% instances with $\\geq 1$ abstain)",
        )
        finite = np.isfinite(aabs_vals)
        ax.scatter(
            x[finite], aabs_vals[finite],
            color="black", marker="o", s=20, zorder=3,
            edgecolors="white", linewidths=0.6,
            label="aabs (% labels skipped, per-cell)",
        )
        for i in range(n_methods):
            if np.isfinite(abs_vals[i]):
                ax.annotate(
                    f"{abs_vals[i]:.1f}",
                    xy=(x[i], abs_vals[i]), xytext=(0, 2),
                    textcoords="offset points", ha="center",
                    fontsize=5, color="black",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m[1] for m in PA_METHODS], rotation=-35, ha="left", fontsize=5,
        )
        ax.set_title(fr"$\alpha={noise}$", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.legend(fontsize=5, loc="upper left", framealpha=0.85, borderpad=0.3)
        ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.4)
        ax.set_ylim(0, max(40, float(np.nanmax(abs_vals)) * 1.15 if np.any(finite) else 40))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0, 0].set_ylabel("abstention rate (%)", fontsize=7)
    axes[1, 0].set_ylabel("abstention rate (%)", fontsize=7)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0, 0.96))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _render_scope(df: pd.DataFrame, out_dir: Path, pa_metric: str, bv_metric: str,
                  title: str, style: str) -> int:
    """Render all 7 metric-dependent chart variants for one scope. Returns count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    render_coverage_risk(df, pa_metric, bv_metric, out_dir / "coverage_risk.pdf", title, style)
    render_coverage_risk_pareto(df, pa_metric, bv_metric, out_dir / "coverage_risk_pareto.pdf", title, style)
    render_paired_bars(df, pa_metric, bv_metric, out_dir / "paired_bars.pdf", title, style)
    render_gain_heatmap(df, pa_metric, bv_metric, out_dir / "gain_heatmap.pdf", title, style)
    render_delta_bars(df, pa_metric, bv_metric, out_dir / "delta_bars.pdf", title, style)
    render_efficiency_quality(df, pa_metric, bv_metric, out_dir / "efficiency_quality.pdf", title, style)
    render_effective_f1(df, pa_metric, bv_metric, out_dir / "effective_f1.pdf", title, style)
    return 7


def build_all(
    long_df: pd.DataFrame,
    out_root: Path,
    style: str,
    pa_metrics: list[str],
    bv_metric: str,
) -> int:
    """Emit charts under <out_root>/<learner>/abstain/<pa_metric>/...

    One subdir per requested PA metric so multiple metrics (e.g. f1_pa and
    jaccard_pa) can coexist for side-by-side comparison.
    """
    n = 0
    for learner in LEARNERS:
        learner_df = long_df[long_df.base_learner == learner]
        if learner_df.empty:
            continue
        # Metric-independent abstention_rate charts (one per (learner, scope)),
        # placed under a shared subdir parallel to the pa_metric dirs.
        shared_root = out_root / learner.lower() / "abstain" / "_shared"
        (shared_root / "aggregate").mkdir(parents=True, exist_ok=True)
        render_abstention_rate(
            learner_df, "", "",
            shared_root / "aggregate" / "abstention_rate.pdf",
            f"{learner} aggregate: abstention rates", style,
        )
        n += 1
        for ds in DATASETS:
            ds_df = learner_df[learner_df.dataset == ds]
            if ds_df.empty:
                continue
            ds_out = shared_root / "per_dataset" / ds
            ds_out.mkdir(parents=True, exist_ok=True)
            render_abstention_rate(
                ds_df, "", "",
                ds_out / "abstention_rate.pdf",
                f"{learner} on {ds}: abstention rates", style,
            )
            n += 1
        # Per-metric charts (coverage_risk, paired_bars, ...).
        for pa_metric in pa_metrics:
            if pa_metric not in learner_df.metric.unique():
                continue
            base = out_root / learner.lower() / "abstain" / pa_metric
            n += _render_scope(
                learner_df, base / "aggregate", pa_metric, bv_metric,
                f"{learner} aggregate ({pa_metric})", style,
            )
            for ds in DATASETS:
                ds_df = learner_df[learner_df.dataset == ds]
                if ds_df.empty:
                    continue
                n += _render_scope(
                    ds_df, base / "per_dataset" / ds, pa_metric, bv_metric,
                    f"{learner} on {ds} ({pa_metric})", style,
                )
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--style", default="enhanced", choices=["original", "enhanced"]
    )
    parser.add_argument(
        "--pa_metrics",
        nargs="+",
        default=["f1_pa", "jaccard_pa"],
        help=(
            "One or more PA-side quality metrics. Each metric becomes its "
            "own subdirectory so multiple metrics coexist for comparison."
        ),
    )
    parser.add_argument(
        "--bv_metric",
        default="f1",
        help="BV-side quality metric used as the no-abstention reference.",
    )
    args = parser.parse_args()

    long_df = load_long_df()
    if long_df.empty:
        raise SystemExit("No data loaded. Check FOLDER_MAP.")

    avail_metrics = set(long_df.metric.unique())
    missing = [m for m in args.pa_metrics if m not in avail_metrics]
    if missing:
        print(
            f"warning: pa_metrics not in summaries (skipping): {missing}. "
            f"Available PA-ish: "
            f"{sorted(m for m in avail_metrics if m.endswith('_pa') or m in ('rec','arec','abs','aabs'))}"
        )

    out_root = Path(args.output_dir)
    n = build_all(long_df, out_root, args.style, args.pa_metrics, args.bv_metric)
    print(
        f"Wrote {n} abstain charts (style={args.style}, "
        f"pa_metrics={args.pa_metrics}, bv_metric={args.bv_metric}) under {out_root}"
    )


if __name__ == "__main__":
    main()
