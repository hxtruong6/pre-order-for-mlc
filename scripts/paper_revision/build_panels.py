#!/usr/bin/env python3
"""Build per-metric bar-chart PDFs for paper_revision/results_revision.tex.

Reads ``results/full_<ds>[_lgbm]_summary/<ds>_<PredictionType>_summary.csv``
files (produced by ``preorder4mlc/utils/summarize_metrics.py``) and emits
one matplotlib bar-chart PDF per (learner, scope, prediction_type, metric):

    paper_revision/figures/<learner>/aggregate/<bv|pa>/<metric>.pdf
    paper_revision/figures/<learner>/per_dataset/<ds>/<bv|pa>/<metric>.pdf

Within a panel: x-axis = method index 1..14 (PA: 1..8). Four bars per
method, colored red/blue/green/black for noise alpha in {0.0, 0.1, 0.2,
0.3} (matches the original paper's caption legend). Methods missing for
a given (dataset, learner) pair are rendered as an empty slot with a
small grey em dash above the axis.

MLkNN is base-learner-independent: rows from RF folders are reused under
LGBM where the LGBM folder has no MLkNN entry.

Usage::

    python scripts/paper_revision/build_panels.py
    python scripts/paper_revision/build_panels.py --output_dir /tmp/figs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUTPUT = REPO_ROOT / "paper_revision" / "figures"

DATASETS = [
    # 6 original-paper datasets (RF runs in results/final_20260514_v2_summary/).
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
    "VirusPseAAC",
]
LEARNERS = ["RF", "LGBM"]
PREDICTION_TYPES = ["BinaryVector", "PartialAbstention", "ScoreVector"]
NOISE_LEVELS = ["0.0", "0.1", "0.2", "0.3"]

# Distinct marker per noise level so figures are readable in greyscale and
# for colour-blind readers (reviewer suggestion).
NOISE_MARKERS = {"0.0": "o", "0.1": "s", "0.2": "D", "0.3": "^"}

# Two palettes, switched by --style:
#   "original" reproduces the qualitative red/blue/green/black scheme used by
#   the original paper (Table 5 caption).
#   "enhanced" uses ColorBrewer Reds[4] so noise level maps to colour intensity
#   (light = clean, dark = noisy), perceptually-ordered and colorblind-safe.
NOISE_COLORS_ORIGINAL = {
    "0.0": "#FF0000",
    "0.1": "#0000FF",
    "0.2": "#339933",
    "0.3": "#000000",
}
NOISE_COLORS_ENHANCED = {
    # ColorBrewer YlOrRd[5] dropping the lightest shade. Wider hue spread
    # than the all-Reds palette so adjacent noise levels are easier to
    # distinguish, while still mapping intensity to higher noise.
    "0.0": "#fecc5c",
    "0.1": "#fd8d3c",
    "0.2": "#f03b20",
    "0.3": "#bd0026",
}

# Per-(dataset, learner) summary folder names.
# The 6 original-paper datasets (+ Yeast) all live in one shared RF folder.
_ORIGINAL_PAPER_FOLDER = "final_20260514_v2_summary"
FOLDER_MAP = {
    # Original-paper datasets — RF only.
    ("GpositivePseAAC", "RF"): _ORIGINAL_PAPER_FOLDER,
    ("emotions", "RF"): _ORIGINAL_PAPER_FOLDER,
    ("scene", "RF"): _ORIGINAL_PAPER_FOLDER,
    ("PlantPseAAC", "RF"): _ORIGINAL_PAPER_FOLDER,
    ("HumanPseAAC", "RF"): _ORIGINAL_PAPER_FOLDER,
    ("Yeast", "RF"): _ORIGINAL_PAPER_FOLDER,
    # LGBM summaries for the 6 original-paper datasets (generated locally
    # from full_<ds>_lgbm_split/ via preorder4mlc.utils.summarize_metrics).
    ("GpositivePseAAC", "LGBM"): "full_gpositivepseaac_lgbm_summary",
    ("emotions", "LGBM"): "full_emotions_lgbm_summary",
    ("scene", "LGBM"): "full_scene_lgbm_summary",
    ("PlantPseAAC", "LGBM"): "full_plantpseaac_lgbm_summary",
    ("HumanPseAAC", "LGBM"): "full_humanpseaac_lgbm_summary",
    ("Yeast", "LGBM"): "full_yeast_lgbm_summary",
    # Revision-extension datasets.
    ("CHD_49", "RF"): _ORIGINAL_PAPER_FOLDER,
    ("CHD_49", "LGBM"): "full_chd_49_lgbm_summary",
    ("Water-quality", "RF"): _ORIGINAL_PAPER_FOLDER,
    ("VirusPseAAC", "RF"): _ORIGINAL_PAPER_FOLDER,
    ("Water-quality", "LGBM"): "full_water_quality_lgbm_summary",
    ("enron", "RF"): "full_enron_split_summary",
}

# Method encoding (1-indexed). PartialAbstention / ScoreVector use the first 8.
METHOD_ORDER = [
    ("PartialOrder__Hamming__2", "PA-H-2"),
    ("PartialOrder__Hamming__None", "PA-H"),
    ("PartialOrder__Subset__2", "PA-S-2"),
    ("PartialOrder__Subset__None", "PA-S"),
    ("PreOrder__Hamming__2", "PR-H-2"),
    ("PreOrder__Hamming__None", "PR-H"),
    ("PreOrder__Subset__2", "PR-S-2"),
    ("PreOrder__Subset__None", "PR-S"),
    ("br", "BR"),
    ("cc", "CC"),
    ("clr", "CLR"),
    ("ecc", "ECC"),
]

PTYPE_DIR = {"BinaryVector": "bv", "PartialAbstention": "pa", "ScoreVector": "sv"}

_CELL_RE = re.compile(r"^([+\-0-9.eE]+)\s*[±+\-/]\s*([+\-0-9.eE]+)$")


def parse_cell(cell) -> float:
    """Parse a ``mean+/-std`` string to its mean. Returns NaN on bad input."""
    if cell is None:
        return float("nan")
    if isinstance(cell, float):
        return cell
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return float("nan")
    m = _CELL_RE.match(s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_long_df() -> pd.DataFrame:
    """Load all per-dataset summaries into one long-format DataFrame.

    Columns: dataset, base_learner, prediction_type, algorithm, metric,
    noise, value (mean only, float).
    """
    rows: list[dict] = []
    for (ds, learner), folder in FOLDER_MAP.items():
        folder_path = RESULTS_DIR / folder
        if not folder_path.exists():
            continue
        for ptype in PREDICTION_TYPES:
            csv = folder_path / f"{ds}_{ptype}_summary.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            for _, row in df.iterrows():
                algo = row["Algorithm"]
                for col in df.columns[1:]:
                    if "__" not in col:
                        continue
                    metric, noise = col.rsplit("__", 1)
                    if noise not in NOISE_LEVELS:
                        continue
                    rows.append(
                        {
                            "dataset": ds,
                            "base_learner": learner,
                            "prediction_type": ptype,
                            "algorithm": algo,
                            "metric": metric,
                            "noise": noise,
                            "value": parse_cell(row[col]),
                        }
                    )
    return pd.DataFrame(rows)


def methods_for_ptype(ptype: str) -> list[tuple[str, str]]:
    # Both PartialAbstention and ScoreVector outputs only exist for the 8
    # order-based predictors (PA/PR). Standard MLC baselines do not emit a
    # probabilistic ScoreVector or partial-abstention prediction.
    if ptype in ("PartialAbstention", "ScoreVector"):
        return METHOD_ORDER[:8]
    return METHOD_ORDER


def build_values_matrix(
    df_subset: pd.DataFrame, methods: list[tuple[str, str]]
) -> np.ndarray:
    """Return a (n_methods, n_noise) matrix of mean values across df_subset.

    df_subset is restricted to one (prediction_type, base_learner, metric)
    and optionally one dataset. Aggregate scope passes multiple datasets;
    we average across them (NaN-aware mean: methods missing for some
    datasets only count where present).
    """
    n_m, n_n = len(methods), len(NOISE_LEVELS)
    values = np.full((n_m, n_n), np.nan)
    if df_subset.empty:
        return values
    for i, (algo, _) in enumerate(methods):
        for j, noise in enumerate(NOISE_LEVELS):
            sel = df_subset[
                (df_subset.algorithm == algo) & (df_subset.noise == noise)
            ]
            if sel.empty:
                continue
            vals = sel.value.dropna().to_numpy()
            if vals.size:
                values[i, j] = float(np.mean(vals))
    return values


def _save_tikz(fig, out_path: Path) -> None:
    """Save the current figure as a matplot2tikz .tex file at out_path."""
    try:
        import matplot2tikz  # lazy: only required when --format includes tikz
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "matplot2tikz is not installed. Install with `pip install matplot2tikz` "
            "or run with --format pdf."
        ) from exc
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matplot2tikz.save(str(out_path), figure=fig)


def render_panel(
    values: np.ndarray,
    methods: list[tuple[str, str]],
    metric: str,
    out_path_pdf: Path | None,
    out_path_tikz: Path | None = None,
    style: str = "original",
    emit_notitle: bool = False,
) -> None:
    """Render a single panel as dotted lines with markers.

    ``style="original"`` reproduces the original paper's look (red/blue/green/
    cyan, numeric xticks 1..N only, fixed ylim [0,100] for bounded metrics).
    ``style="enhanced"`` uses ColorBrewer Reds[4], a 2-tier xtick row
    (number on top, method label below), and auto-zoomed y-axis.
    """
    n_methods = len(methods)
    palette = NOISE_COLORS_ENHANCED if style == "enhanced" else NOISE_COLORS_ORIGINAL
    fig_w = 3.6 if style == "enhanced" else 3.5
    fig_h = 2.4 if style == "enhanced" else 2.1
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x_pos = np.arange(1, n_methods + 1, dtype=float)
    # Most metrics are bounded in [0,1] so we render as percent. ScoreVector's
    # `coverage` is unbounded; fall back to raw values when max > 1.5.
    finite_vals = values[np.isfinite(values)]
    use_percent = finite_vals.size == 0 or float(np.nanmax(finite_vals)) <= 1.5
    scale = 100.0 if use_percent else 1.0

    # Method groups (1-based, inclusive ranges) — each group is a distinct
    # method family so the dotted line is broken between groups (insert NaN
    # at boundary so matplotlib doesn't connect across families).
    #   PA:        1..4
    #   PR:        5..8
    #   Std base:  9..11   (BR, CC, CLR)
    #   Extended:  12      (ECC)
    group_ranges = [(1, 4), (5, 8), (9, 11), (12, 12)]
    # Drop ranges that fall entirely outside this panel's method count.
    group_ranges = [(lo, hi) for lo, hi in group_ranges if lo <= n_methods]
    group_ranges = [(lo, min(hi, n_methods)) for lo, hi in group_ranges]

    # Jitter markers per noise level on the x-axis so red/blue/green/black
    # dots at the same method don't completely overlap.
    jitter_step = 0.12
    centered = (np.arange(len(NOISE_LEVELS)) - (len(NOISE_LEVELS) - 1) / 2.0)
    noise_jitters = {n: float(centered[i] * jitter_step)
                     for i, n in enumerate(NOISE_LEVELS)}
    for i, noise in enumerate(NOISE_LEVELS):
        y_raw = values[:, i] * scale
        jx = noise_jitters[noise]
        # Build (x, y) sequences with NaN gaps between groups so the dotted
        # line breaks at each family boundary.
        x_segs: list[float] = []
        y_segs: list[float] = []
        for g_idx, (lo, hi) in enumerate(group_ranges):
            if g_idx > 0:
                x_segs.append(np.nan)
                y_segs.append(np.nan)
            for j in range(lo, hi + 1):
                x_segs.append(float(j) + jx)
                y_segs.append(float(y_raw[j - 1]))
        ax.plot(
            x_segs,
            y_segs,
            linestyle=(0, (2, 2)),
            linewidth=0.9,
            marker=NOISE_MARKERS[noise],
            markersize={"o": 4.1, "s": 3.5, "D": 3.2, "^": 4.1}[NOISE_MARKERS[noise]],
            color=palette[noise],
            markeredgewidth=0.5,
            markeredgecolor="white",
            zorder=3,
        )
    # Vertical dividers grouping methods into 4 blocks:
    #   1..4   = Partial-order predictors (PA-*)
    #   5..8   = Pre-order predictors (PR-*)
    #   9..11  = Standard MLC baselines (BR, CC, CLR)
    #   12     = Extended baseline (ECC)
    # PA/PR boundary is always drawn; baseline boundaries only when those
    # methods appear in this panel (PA / SV panels show only 1..8).
    ax.axvline(x=4.5, color="grey", linewidth=0.5, linestyle="--", alpha=0.5, zorder=1)
    if n_methods > 8:
        ax.axvline(x=8.5, color="grey", linewidth=0.6, linestyle="--", alpha=0.65, zorder=1)
    if n_methods > 11:
        ax.axvline(x=11.5, color="grey", linewidth=0.6, linestyle="--", alpha=0.65, zorder=1)
    # Methods that have no data anywhere in this panel get a small grey "x"
    # at the bottom so the reader can tell "missing" from "low value".
    all_nan_methods = np.all(np.isnan(values), axis=1)
    for j in np.where(all_nan_methods)[0]:
        ax.plot(
            x_pos[j], 0,
            marker="x", markersize=4.0,
            color="lightgrey", markeredgewidth=0.8, linestyle="None",
        )
    ax.set_xticks(x_pos)
    ax2 = None
    if style == "enhanced":
        # 2-tier xticks: number on top, method label rotated below.
        ax.set_xticklabels([str(int(p)) for p in x_pos], fontsize=9)
        # Secondary tick row beneath: method labels.
        labels = [m[1] for m in methods]
        ax2 = ax.secondary_xaxis("bottom")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, fontsize=5, rotation=-35, ha="left")
        ax2.tick_params(axis="x", pad=10, length=0)
        ax2.spines["bottom"].set_visible(False)
    else:
        ax.set_xticklabels([str(int(p)) for p in x_pos], fontsize=9)
    ax.set_xlim(0.7, n_methods + 0.3)
    # Y-axis: always auto-zoom to the data range (this matches the original
    # paper's Table 5 — see the GpositivePse / plantPse / HumanPse panels
    # where the y-axis is tight around the data, not fixed [0,100]).
    if finite_vals.size:
        ymin = float(np.nanmin(finite_vals)) * scale
        ymax = float(np.nanmax(finite_vals)) * scale
        span = max(ymax - ymin, 1.0)
        pad = span * 0.10
        # Always pad above the max so markers sitting at or near 100 are
        # not clipped against the top spine. Bottom is still floored at 0.
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    elif use_percent:
        ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=9)
    # Per-panel title (metric name) is needed by the per-dataset appendix
    # tables to identify each subplot. Reduced from 8pt -> 7pt + tight
    # padding so its bbox is smaller and trim is more predictable.
    title_obj = ax.set_title(metric, fontsize=7, pad=2.0)
    ax.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.40)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    if out_path_pdf is not None:
        out_path_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02
        )
        # Also emit a no-title variant for paper-results tables, which
        # carry their own LaTeX caption beneath each panel. The x-axis is
        # also re-labelled with method names (PA-H-2, PR-S, BR, ...) so
        # the reader doesn't need the table caption's encoding legend.
        # Only emit for metrics referenced by paper-results / G2-G7 to
        # keep total file count under Overleaf's 2000-file limit.
        if emit_notitle:
            title_obj.set_visible(False)
            method_names = [m[1] for m in methods]
            ax.set_xticklabels(method_names, rotation=35, ha="right", rotation_mode="anchor", fontsize=8 if n_methods > 8 else 9)
            # Hide the secondary method-label axis so it doesn't draw a second,
            # smaller copy of the method names on top of the primary labels.
            if ax2 is not None:
                ax2.set_visible(False)
            fig.tight_layout(pad=0.2)
            notitle_path = out_path_pdf.with_name(out_path_pdf.stem + "_notitle.pdf")
            fig.savefig(
                notitle_path, format="pdf", bbox_inches="tight", pad_inches=0.02
            )
    if out_path_tikz is not None:
        _save_tikz(fig, out_path_tikz)
    plt.close(fig)


def _panel_paths(
    out_root: Path,
    learner: str,
    scope_parts: list[str],
    ptype: str,
    metric: str,
    formats: set[str],
) -> tuple[Path | None, Path | None]:
    """Return (pdf_path, tikz_path) for the requested formats (None if disabled)."""
    base = out_root / learner.lower()
    for part in scope_parts:
        base = base / part
    base = base / PTYPE_DIR[ptype]
    pdf = base / f"{metric}.pdf" if "pdf" in formats else None
    tikz = base / f"{metric}.tex" if "tikz" in formats else None
    return pdf, tikz


def build_all(
    long_df: pd.DataFrame, out_root: Path, formats: set[str], style: str
) -> int:
    """Emit every panel under the requested formats. Returns figure count.

    LGBM only emits aggregate panels (RF keeps both aggregate + per-dataset)
    to keep the LGBM appendix focused on cross-dataset trend.
    """
    n_panels = 0
    for learner in LEARNERS:
        learner_df = long_df[long_df.base_learner == learner]
        for ptype in PREDICTION_TYPES:
            ptype_df = learner_df[learner_df.prediction_type == ptype]
            if ptype_df.empty:
                continue
            methods = methods_for_ptype(ptype)
            metrics = sorted(ptype_df.metric.unique())

            # Aggregate (mean across datasets).
            for metric in metrics:
                sub = ptype_df[ptype_df.metric == metric]
                values = build_values_matrix(sub, methods)
                pdf, tikz = _panel_paths(
                    out_root, learner, ["aggregate"], ptype, metric, formats
                )
                render_panel(values, methods, metric, pdf, tikz, style=style)
                n_panels += 1

            # Per-dataset only for RF (LGBM is aggregate-only).
            if learner != "RF":
                continue
            # _notitle variants are only needed for metrics referenced by
            # paper-results tables (3.x) and appendix G2-G7. Restricting
            # keeps total file count under Overleaf's 2000-file limit.
            paper_metrics = {
                "BinaryVector":      {"f1", "hamming_accuracy", "subset0_1",
                                       "jaccard", "macro_f1", "micro_f1",
                                       "afrd", "mfrd"},
                "PartialAbstention": {"f1_pa", "hamming_accuracy_pa",
                                       "subset0_1_pa", "jaccard_pa",
                                       "macro_f1_pa", "micro_f1_pa",
                                       "aabs", "abs"},
                "ScoreVector":       set(),
            }
            for ds in DATASETS:
                ds_sub = ptype_df[ptype_df.dataset == ds]
                if ds_sub.empty:
                    continue
                for metric in metrics:
                    sub = ds_sub[ds_sub.metric == metric]
                    values = build_values_matrix(sub, methods)
                    pdf, tikz = _panel_paths(
                        out_root,
                        learner,
                        ["per_dataset", ds],
                        ptype,
                        metric,
                        formats,
                    )
                    render_panel(
                        values, methods, metric, pdf, tikz, style=style,
                        emit_notitle=metric in paper_metrics.get(ptype, set()),
                    )
                    n_panels += 1
    return n_panels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT),
        help="Where to write panels (default: paper_revision/figures).",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "tikz", "both"],
        help=(
            "Output format(s). 'pdf' (default) writes <metric>.pdf for "
            "\\includegraphics. 'tikz' writes <metric>.tex via matplot2tikz "
            "for \\input{}. 'both' emits both side by side."
        ),
    )
    parser.add_argument(
        "--style",
        default="original",
        choices=["original", "enhanced"],
        help=(
            "Visual style. 'original' reproduces the paper's Table 5 look "
            "(red/blue/green/black, numeric xticks, ylim [0,100]). 'enhanced' "
            "uses ColorBrewer Reds for noise, 2-tier xticks with method "
            "labels, and auto-zoomed y-axis."
        ),
    )
    args = parser.parse_args()
    out_root = Path(args.output_dir)

    if args.format == "both":
        formats: set[str] = {"pdf", "tikz"}
    else:
        formats = {args.format}

    print(f"Reading summary CSVs from {RESULTS_DIR}")
    long_df = load_long_df()
    if long_df.empty:
        raise SystemExit("No summary CSVs loaded. Check FOLDER_MAP paths.")
    print(f"  loaded {len(long_df):,} rows from {len(FOLDER_MAP)} folders")

    n = build_all(long_df, out_root, formats, style=args.style)
    fmts = "+".join(sorted(formats))
    print(f"Wrote {n} panels ({fmts}, style={args.style}) under {out_root}")


if __name__ == "__main__":
    main()
