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
    baseline_qualities: dict[str, float] = {}
    for algo, _label in BASELINE_METHODS:
        # average across noise levels for the reference line
        vals = df[(df.algorithm == algo) & (df.metric == bv_metric)].value.dropna()
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
            label=f"best baseline {bv_metric}",
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

    ax.set_xlabel("Abstention rate (%)", fontsize=7)
    ax.set_ylabel(f"{pa_metric} on retained (%)", fontsize=7)
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
    noise: str = "0.0",
) -> None:
    """Render BV-vs-PA paired bars for all 12 methods at one noise level."""
    palette = NOISE_COLORS_ENHANCED if style == "enhanced" else NOISE_COLORS_ORIGINAL
    bv_color = "#9ecae1"  # light blue for standard MLC
    pa_color = palette[noise]

    n_methods = len(METHOD_ORDER)
    bv_vals = np.full(n_methods, np.nan)
    pa_vals = np.full(n_methods, np.nan)
    for i, (algo, _) in enumerate(METHOD_ORDER):
        bv_vals[i] = _aggregate_value(df, algo, bv_metric, noise) * 100.0
        if i < 8:  # only order-based methods abstain
            pa_vals[i] = _aggregate_value(df, algo, pa_metric, noise) * 100.0

    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    x = np.arange(n_methods)
    width = 0.4
    ax.bar(x - width / 2, bv_vals, width, color=bv_color, label=f"BV {bv_metric}")
    ax.bar(x + width / 2, pa_vals, width, color=pa_color, label=f"PA {pa_metric}")

    # Annotate gap above PA bar for the 8 PA methods that have both
    for i in range(8):
        if np.isfinite(bv_vals[i]) and np.isfinite(pa_vals[i]):
            gap = pa_vals[i] - bv_vals[i]
            if gap > 0:
                ax.annotate(
                    f"+{gap:.1f}",
                    xy=(x[i] + width / 2, pa_vals[i]),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=5,
                    color="darkgreen",
                )

    ax.axvline(x=7.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in METHOD_ORDER], rotation=-35, ha="left", fontsize=5)
    ax.set_ylabel("score (%)", fontsize=7)
    ax.set_title(fr"{title}  ($\alpha={noise}$)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=5, loc="lower left", framealpha=0.85, borderpad=0.3)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_all(
    long_df: pd.DataFrame,
    out_root: Path,
    style: str,
    pa_metric: str,
    bv_metric: str,
) -> int:
    n = 0
    for learner in LEARNERS:
        learner_df = long_df[long_df.base_learner == learner]
        if learner_df.empty:
            continue

        # Aggregate (mean across datasets, computed by pooling all rows).
        agg_dir = out_root / learner.lower() / "abstain" / "aggregate"
        title = f"{learner} aggregate"
        render_coverage_risk(
            learner_df, pa_metric, bv_metric,
            agg_dir / "coverage_risk.pdf", title, style,
        )
        render_paired_bars(
            learner_df, pa_metric, bv_metric,
            agg_dir / "paired_bars.pdf", title, style,
        )
        n += 2

        # Per-dataset.
        for ds in DATASETS:
            ds_df = learner_df[learner_df.dataset == ds]
            if ds_df.empty:
                continue
            ds_dir = out_root / learner.lower() / "abstain" / "per_dataset" / ds
            t = f"{learner} on {ds}"
            render_coverage_risk(
                ds_df, pa_metric, bv_metric,
                ds_dir / "coverage_risk.pdf", t, style,
            )
            render_paired_bars(
                ds_df, pa_metric, bv_metric,
                ds_dir / "paired_bars.pdf", t, style,
            )
            n += 2
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--style", default="enhanced", choices=["original", "enhanced"]
    )
    parser.add_argument(
        "--pa_metric",
        default="f1_pa",
        help=(
            "PA-side quality metric. Default f1_pa is available in current "
            "summaries; switch to jaccard_pa once eval is re-run with the "
            "extended PA metric set."
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

    # Sanity check the requested metric exists somewhere.
    if args.pa_metric not in long_df.metric.unique():
        avail = sorted(
            m for m in long_df.metric.unique() if m.endswith("_pa") or m in ("rec", "arec", "abs", "aabs")
        )
        raise SystemExit(
            f"pa_metric {args.pa_metric!r} not found in summaries. "
            f"Available PA metrics: {avail}"
        )

    out_root = Path(args.output_dir)
    n = build_all(long_df, out_root, args.style, args.pa_metric, args.bv_metric)
    print(
        f"Wrote {n} abstain charts (style={args.style}, "
        f"pa_metric={args.pa_metric}, bv_metric={args.bv_metric}) under {out_root}"
    )


if __name__ == "__main__":
    main()
