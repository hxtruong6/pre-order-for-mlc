#!/usr/bin/env python3
"""Generate 3 styling versions of sample panels for visual comparison.

Outputs to paper_revision/figures_compare/{v1,v2,v3}/<panel>.pdf

v1 — High impact:   per-shape size correction + wider jitter + in-panel legend
v2 — + Medium:      v1 + black edge + loose dot pattern + tight y-axis
v3 — + Low:         v2 + reduced grid opacity + thinner linewidth

Run:
    python scripts/paper_revision/compare_versions.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "paper_revision"))

from build_panels import (  # noqa: E402
    FOLDER_MAP,
    METHOD_ORDER,
    NOISE_COLORS_ORIGINAL,
    NOISE_LEVELS,
    NOISE_MARKERS,
    PREDICTION_TYPES,
    RESULTS_DIR,
    load_long_df,
    methods_for_ptype,
    build_values_matrix,
)

OUT_ROOT = REPO_ROOT / "paper_revision" / "figures_compare"

# Sample panels: (dataset, learner, ptype, metric, title_suffix)
SAMPLES = [
    ("emotions",        "RF", "PartialAbstention", "aabs",            "emotions · PA · aabs"),
    ("GpositivePseAAC", "RF", "PartialAbstention", "aabs",            "GpositivePseAAC · PA · aabs"),
    ("emotions",        "RF", "BinaryVector",       "hamming_accuracy","emotions · BV · hamming_acc"),
    ("CHD_49",          "RF", "PartialAbstention",  "aabs",            "CHD_49 · PA · aabs"),
]

# ── shared constants ──────────────────────────────────────────────────────────
PALETTE = NOISE_COLORS_ORIGINAL
GROUP_RANGES = [(1, 4), (5, 8), (9, 11), (12, 12)]


def _jitter_dict(jitter_step: float) -> dict[str, float]:
    centered = np.arange(len(NOISE_LEVELS)) - (len(NOISE_LEVELS) - 1) / 2.0
    return {n: float(centered[i] * jitter_step) for i, n in enumerate(NOISE_LEVELS)}


def _add_legend(ax, marker_size: float, edge_color: str, edge_width: float) -> None:
    alpha_labels = {"0.0": "α=0.0", "0.1": "α=0.1", "0.2": "α=0.2", "0.3": "α=0.3"}
    handles = [
        Line2D([0], [0],
               marker=NOISE_MARKERS[n],
               color=PALETTE[n],
               linestyle="None",
               markersize=marker_size,
               markeredgewidth=edge_width,
               markeredgecolor=edge_color,
               label=alpha_labels[n])
        for n in NOISE_LEVELS
    ]
    ax.legend(handles=handles, fontsize=5, loc="upper right",
              framealpha=0.7, borderpad=0.4, handlelength=1.0,
              handletextpad=0.4, labelspacing=0.3)


def render_panel(
    values: np.ndarray,
    methods: list[tuple[str, str]],
    metric: str,
    title: str,
    out_path: Path,
    # styling knobs
    marker_sizes: dict[str, float],  # per-marker shape size
    jitter_step: float,
    show_legend: bool,
    edge_color: str,
    edge_width: float,
    linestyle,
    tight_yaxis: bool,
    grid_alpha: float,
    linewidth: float,
) -> None:
    n_methods = len(methods)
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    x_pos = np.arange(1, n_methods + 1, dtype=float)

    finite_vals = values[np.isfinite(values)]
    use_percent = finite_vals.size == 0 or float(np.nanmax(finite_vals)) <= 1.5
    scale = 100.0 if use_percent else 1.0

    gr = [(lo, min(hi, n_methods)) for lo, hi in GROUP_RANGES if lo <= n_methods]
    noise_jitters = _jitter_dict(jitter_step)

    for noise in NOISE_LEVELS:
        i = NOISE_LEVELS.index(noise)
        y_raw = values[:, i] * scale
        jx = noise_jitters[noise]
        mk = NOISE_MARKERS[noise]
        ms = marker_sizes[mk]

        x_segs: list[float] = []
        y_segs: list[float] = []
        for g_idx, (lo, hi) in enumerate(gr):
            if g_idx > 0:
                x_segs.append(np.nan)
                y_segs.append(np.nan)
            for j in range(lo, hi + 1):
                x_segs.append(float(j) + jx)
                y_segs.append(float(y_raw[j - 1]))

        ax.plot(
            x_segs, y_segs,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=mk,
            markersize=ms,
            color=PALETTE[noise],
            markeredgewidth=edge_width,
            markeredgecolor=edge_color,
            zorder=3,
        )

    # Vertical dividers
    ax.axvline(x=4.5, color="grey", linewidth=0.5, linestyle="--", alpha=0.5, zorder=1)
    if n_methods > 8:
        ax.axvline(x=8.5, color="grey", linewidth=0.6, linestyle="--", alpha=0.65, zorder=1)
    if n_methods > 11:
        ax.axvline(x=11.5, color="grey", linewidth=0.6, linestyle="--", alpha=0.65, zorder=1)

    # Missing-data markers
    all_nan = np.all(np.isnan(values), axis=1)
    for j in np.where(all_nan)[0]:
        ax.plot(x_pos[j], 0, marker="x", markersize=4.0,
                color="lightgrey", markeredgewidth=0.8, linestyle="None")

    ax.set_xticks(x_pos)
    method_names = [m[1] for m in methods]
    ax.set_xticklabels(method_names, rotation=-35, ha="left", fontsize=9)
    ax.set_xlim(0.7, n_methods + 0.3)

    # Y-axis
    if finite_vals.size:
        ymin = float(np.nanmin(finite_vals)) * scale
        ymax = float(np.nanmax(finite_vals)) * scale
        span = max(ymax - ymin, 1.0)
        pad = span * 0.10
        low = (max(0.0, ymin - pad) if not tight_yaxis else ymin - pad)
        ax.set_ylim(low, ymax + pad)
    elif use_percent:
        ax.set_ylim(0, 100)

    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(f"{metric}\n({title})", fontsize=7, pad=2.0)
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=grid_alpha)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if show_legend:
        _add_legend(ax, max(marker_sizes.values()), edge_color, edge_width)

    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ── version configs ───────────────────────────────────────────────────────────
VERSIONS = {
    "v1_high": dict(
        marker_sizes={"o": 3.8, "s": 3.2, "D": 3.0, "^": 3.8},
        jitter_step=0.12,
        show_legend=True,
        edge_color="white",
        edge_width=0.5,
        linestyle=":",
        tight_yaxis=False,
        grid_alpha=0.4,
        linewidth=0.8,
    ),
    "v2_high_medium": dict(
        marker_sizes={"o": 3.8, "s": 3.2, "D": 3.0, "^": 3.8},
        jitter_step=0.12,
        show_legend=True,
        edge_color="black",
        edge_width=0.3,
        linestyle=(0, (2, 3)),
        tight_yaxis=True,
        grid_alpha=0.4,
        linewidth=0.8,
    ),
    "v3_all": dict(
        marker_sizes={"o": 3.8, "s": 3.2, "D": 3.0, "^": 3.8},
        jitter_step=0.12,
        show_legend=True,
        edge_color="black",
        edge_width=0.3,
        linestyle=(0, (2, 3)),
        tight_yaxis=True,
        grid_alpha=0.25,
        linewidth=0.7,
    ),
}


def main() -> None:
    print("Loading data...")
    df = load_long_df()

    for ver_name, cfg in VERSIONS.items():
        print(f"\n── {ver_name} ──")
        for ds, learner, ptype, metric, label in SAMPLES:
            methods = methods_for_ptype(ptype)
            sub = df[
                (df.dataset == ds)
                & (df.base_learner == learner)
                & (df.prediction_type == ptype)
                & (df.metric == metric)
            ]
            values = build_values_matrix(sub, methods)
            out = OUT_ROOT / ver_name / f"{ds}_{ptype}_{metric}.pdf"
            render_panel(values, methods, metric, label, out, **cfg)
            print(f"  wrote {out.name}")

    print(f"\nDone. Output → {OUT_ROOT}")
    print("Open all PNGs for side-by-side comparison:")
    pngs = sorted(OUT_ROOT.rglob("*.png"))
    # Print grouped by sample
    for ds, _, ptype, metric, _ in SAMPLES:
        name = f"{ds}_{ptype}_{metric}.png"
        matches = [p for p in pngs if p.name == name]
        print(f"  {name}: {' | '.join(str(p) for p in matches)}")


if __name__ == "__main__":
    main()
