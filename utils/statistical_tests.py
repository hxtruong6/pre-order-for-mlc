"""
Statistical significance testing across datasets for the BOPOs paper revision.

For each (metric, noise_rate, prediction_type) we:
  - build an [algorithms x datasets] matrix of means parsed from the per-dataset
    `mean+/-std` summary CSVs,
  - run Friedman's chi-square test,
  - run Nemenyi post-hoc (studentized range distribution),
  - compute average ranks (direction depends on whether metric is loss-like),
  - compute win/tie/loss counts of each BOPOs variant vs each baseline,
  - render a Critical-Difference diagram (Demsar 2006) per metric/noise.

Run:
    python utils/statistical_tests.py \
        --results_dir results/final_0624_summary \
        --output_dir results/final_0624_summary/stats
"""

from __future__ import annotations

import argparse
import os
import re
from glob import glob
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range


# --- Metric semantics -------------------------------------------------------
# Verified against evaluation_metric.py: subset0_1 in the summary CSVs is the
# exact-match accuracy (predicted == true), so higher is better.
HIGHER_IS_BETTER = {
    "hamming_accuracy",
    "f1",
    "jaccard",
    "subset0_1",
    "macro_f1",
    "micro_f1",
    "macro_precision",
    "micro_precision",
    "macro_recall",
    "micro_recall",
    "example_precision",
    "example_recall",
    "hamming_accuracy_pa",
    "f1_pa",
    "subset0_1_pa",
    "arec",
    "rec",
    "hamming_accuracy_PRE_ORDER",
    "hamming_accuracy_PARTIAL_ORDER",
}
LOWER_IS_BETTER = {
    "afrd",
    "mfrd",
    "aabs",
    "abs",
    "subset0_1_PRE_ORDER",
    "subset0_1_PARTIAL_ORDER",
}

BASELINES = {"br", "cc", "clr"}
TIE_TOL = 1e-6

CELL_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*[±+/-]+")


def parse_mean(cell) -> float:
    """Parse '0.1417+/-0.0243' (the unicode +/- 0xb1 char) and similar."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.nan
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return np.nan
    # Split on unicode +/- 0xb1, ASCII '+-' or '+/-'
    for sep in ("±", "+/-", "+-"):
        if sep in s:
            try:
                return float(s.split(sep)[0])
            except ValueError:
                return np.nan
    try:
        return float(s)
    except ValueError:
        m = CELL_RE.match(s)
        return float(m.group(1)) if m else np.nan


def metric_direction(metric: str) -> int:
    """+1 if higher is better, -1 if lower is better."""
    if metric in HIGHER_IS_BETTER:
        return 1
    if metric in LOWER_IS_BETTER:
        return -1
    # Default: assume higher is better, but warn.
    print(f"[warn] unknown metric '{metric}', defaulting to higher-is-better")
    return 1


# --- Loading ----------------------------------------------------------------

SUMMARY_RE = re.compile(r"^(?P<dataset>.+)_(?P<ptype>BinaryVector|PartialAbstention)_summary\.csv$")


def discover_summaries(results_dir: str) -> List[Tuple[str, str, str]]:
    out = []
    for path in sorted(glob(os.path.join(results_dir, "*_summary.csv"))):
        m = SUMMARY_RE.match(os.path.basename(path))
        if m:
            out.append((path, m.group("dataset"), m.group("ptype")))
    return out


def load_long(summaries: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """Return long DataFrame with columns: dataset, ptype, algorithm, metric, noise, mean."""
    rows = []
    for path, dataset, ptype in summaries:
        df = pd.read_csv(path)
        if "Algorithm" not in df.columns:
            continue
        for _, row in df.iterrows():
            alg = str(row["Algorithm"]).strip()
            if not alg or alg.lower() == "nan":
                continue
            for col in df.columns:
                if col == "Algorithm" or "__" not in col:
                    continue
                metric, noise = col.rsplit("__", 1)
                mean = parse_mean(row[col])
                rows.append((dataset, ptype, alg, metric, noise, mean))
    return pd.DataFrame(
        rows, columns=["dataset", "ptype", "algorithm", "metric", "noise", "mean"]
    )


# --- Stats ------------------------------------------------------------------


def rank_matrix(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Per-dataset (row) ranking of algorithms (columns). Rank 1 = best."""
    # Flip sign for higher-is-better so smaller rank means better.
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


def nemenyi_pvalues(ranks: np.ndarray, n_datasets: int) -> np.ndarray:
    """Pairwise Nemenyi p-values from per-dataset ranks (datasets x algorithms).

    The Nemenyi test statistic is q = (R_i - R_j) / sqrt(k(k+1)/(6N)),
    distributed as a studentized range with k groups and infinite df.
    """
    k = ranks.shape[1]
    avg_ranks = np.nanmean(ranks, axis=0)
    se = np.sqrt(k * (k + 1) / (6.0 * n_datasets))
    pvals = np.ones((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            q = abs(avg_ranks[i] - avg_ranks[j]) / se
            # studentized_range.sf takes q*sqrt(2) under the usual parameterization
            # used by Demsar. scipy's studentized_range uses the standard def with
            # k groups and df=inf; the Nemenyi statistic is the same q.
            pvals[i, j] = studentized_range.sf(q * np.sqrt(2), k, np.inf)
    return pvals


def critical_difference(k: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Demsar (2006) CD = q_alpha * sqrt(k(k+1)/(6N))."""
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_datasets))


def win_tie_loss(
    values: pd.DataFrame, higher_is_better: bool, tol: float = TIE_TOL
) -> pd.DataFrame:
    """For each pair (proposed BOPOs row, baseline col) return W/T/L over datasets."""
    proposed = [a for a in values.columns if a not in BASELINES]
    baselines = [a for a in values.columns if a in BASELINES]
    rec = []
    for p in proposed:
        for b in baselines:
            diff = values[p] - values[b]
            if not higher_is_better:
                diff = -diff
            valid = diff.dropna()
            wins = int((valid > tol).sum())
            losses = int((valid < -tol).sum())
            ties = int(valid.shape[0] - wins - losses)
            rec.append((p, b, wins, ties, losses, valid.shape[0]))
    return pd.DataFrame(rec, columns=["proposed", "baseline", "wins", "ties", "losses", "n"])


# --- CD diagram -------------------------------------------------------------


def plot_cd_diagram(
    avg_ranks: np.ndarray,
    names: List[str],
    cd: float,
    title: str,
    out_path: str,
) -> None:
    """Demsar-style CD diagram. Lower rank = better, drawn on top."""
    k = len(names)
    order = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[order]
    sorted_names = [names[i] for i in order]

    lo = int(np.floor(sorted_ranks.min()))
    hi = int(np.ceil(sorted_ranks.max()))
    if hi == lo:
        hi = lo + 1
    width = hi - lo

    fig_w = max(8.0, 1.2 * k)
    fig, ax = plt.subplots(figsize=(fig_w, 0.35 * k + 2.5))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(-0.5 * k - 1.5, 1.5)
    ax.axis("off")

    # Top axis ticks
    y_axis = 0.0
    ax.plot([lo, hi], [y_axis, y_axis], color="black", lw=1.0)
    for tick in range(lo, hi + 1):
        ax.plot([tick, tick], [y_axis, y_axis + 0.2], color="black", lw=1.0)
        ax.text(tick, y_axis + 0.35, str(tick), ha="center", va="bottom", fontsize=9)

    # Method labels: half on left (best ranks), half on right
    half = (k + 1) // 2
    for idx, (rank, name) in enumerate(zip(sorted_ranks, sorted_names)):
        if idx < half:
            y = -0.5 * (idx + 1) - 0.3
            ax.plot([rank, rank], [y_axis, y], color="black", lw=0.8)
            ax.plot([rank, lo - 0.3], [y, y], color="black", lw=0.8)
            ax.text(
                lo - 0.4, y, f"{name}  ({rank:.2f})", ha="right", va="center", fontsize=9
            )
        else:
            y = -0.5 * (k - idx) - 0.3
            ax.plot([rank, rank], [y_axis, y], color="black", lw=0.8)
            ax.plot([rank, hi + 0.3], [y, y], color="black", lw=0.8)
            ax.text(
                hi + 0.4, y, f"({rank:.2f})  {name}", ha="left", va="center", fontsize=9
            )

    # CD bar
    cd_y = y_axis + 0.9
    bar_left = lo
    bar_right = min(hi, lo + cd)
    ax.plot([bar_left, bar_right], [cd_y, cd_y], color="black", lw=2.0)
    ax.plot([bar_left, bar_left], [cd_y - 0.1, cd_y + 0.1], color="black", lw=2.0)
    ax.plot([bar_right, bar_right], [cd_y - 0.1, cd_y + 0.1], color="black", lw=2.0)
    ax.text((bar_left + bar_right) / 2, cd_y + 0.15, f"CD = {cd:.3f}", ha="center", fontsize=9)

    # Cliques: groups of methods whose pairwise rank diff <= CD
    cliques = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and (sorted_ranks[j + 1] - sorted_ranks[i]) <= cd + 1e-9:
            j += 1
        if j > i:
            cliques.append((i, j))
        i += 1
    # Deduplicate non-maximal cliques
    maximal = []
    for a, b in cliques:
        dominated = any(a2 <= a and b2 >= b and (a2, b2) != (a, b) for a2, b2 in cliques)
        if not dominated:
            maximal.append((a, b))

    base_y = -0.15
    step = 0.12
    for ci, (a, b) in enumerate(maximal):
        y = base_y - ci * step
        ax.plot(
            [sorted_ranks[a] - 0.05, sorted_ranks[b] + 0.05],
            [y, y],
            color="red",
            lw=3.0,
            solid_capstyle="butt",
        )

    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --- Main pipeline ----------------------------------------------------------


def run(results_dir: str, output_dir: str, alpha: float = 0.05) -> None:
    os.makedirs(output_dir, exist_ok=True)
    summaries = discover_summaries(results_dir)
    if not summaries:
        raise SystemExit(f"No *_summary.csv found in {results_dir}")
    long_df = load_long(summaries)

    summary_rows: List[Dict] = []
    wtl_rows: List[Dict] = []
    focus_friedman: Dict[Tuple[str, str], float] = {}

    for (ptype, metric, noise), grp in long_df.groupby(["ptype", "metric", "noise"]):
        pivot = grp.pivot_table(
            index="dataset", columns="algorithm", values="mean", aggfunc="mean"
        )
        # Drop algorithms with too many NaNs (keep those present in >= half datasets)
        valid_alg = pivot.columns[pivot.notna().sum(axis=0) >= max(2, len(pivot) // 2)]
        pivot = pivot[valid_alg]
        # Drop datasets with any NaN remaining for a clean Friedman test
        complete = pivot.dropna(axis=0, how="any")
        n_datasets, k = complete.shape
        if n_datasets < 2 or k < 3:
            continue

        higher = metric_direction(metric) == 1
        try:
            chi2, p_val = friedmanchisquare(
                *[complete[col].values for col in complete.columns]
            )
        except ValueError:
            chi2, p_val = np.nan, np.nan

        summary_rows.append(
            {
                "ptype": ptype,
                "metric": metric,
                "noise": noise,
                "friedman_chi2": chi2,
                "friedman_p": p_val,
                "n_datasets": n_datasets,
                "n_algorithms": k,
            }
        )
        if metric == "hamming_accuracy" and noise in {"0.0", "0.3"} and ptype == "BinaryVector":
            focus_friedman[(metric, noise)] = p_val

        ranks = rank_matrix(complete.values, higher_is_better=higher)
        avg_ranks = np.nanmean(ranks, axis=0)
        algs = list(complete.columns)

        # Average ranks CSV
        rank_df = pd.DataFrame({"algorithm": algs, "avg_rank": avg_ranks}).sort_values(
            "avg_rank"
        )
        rank_path = os.path.join(
            output_dir, f"avg_ranks_{ptype}_{metric}_{noise}.csv"
        )
        rank_df.to_csv(rank_path, index=False)

        # Nemenyi pairwise
        pvals = nemenyi_pvalues(ranks, n_datasets=n_datasets)
        nemenyi_df = pd.DataFrame(pvals, index=algs, columns=algs)
        nemenyi_path = os.path.join(
            output_dir, f"nemenyi_{ptype}_{metric}_{noise}.csv"
        )
        nemenyi_df.to_csv(nemenyi_path)

        # CD diagram
        cd = critical_difference(k, n_datasets, alpha=alpha)
        title = (
            f"CD diagram - {ptype} - {metric} - noise {noise} "
            f"(N={n_datasets}, k={k}, alpha={alpha})"
        )
        cd_path = os.path.join(output_dir, f"cd_{ptype}_{metric}_{noise}.pdf")
        plot_cd_diagram(avg_ranks, algs, cd, title, cd_path)

        # Win/tie/loss
        wtl = win_tie_loss(complete, higher_is_better=higher)
        wtl["ptype"] = ptype
        wtl["metric"] = metric
        wtl["noise"] = noise
        wtl_rows.append(wtl)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["ptype", "metric", "noise"]
    )
    summary_df.to_csv(os.path.join(output_dir, "significance_summary.csv"), index=False)

    if wtl_rows:
        all_wtl = pd.concat(wtl_rows, ignore_index=True)
        all_wtl.to_csv(os.path.join(output_dir, "win_tie_loss.csv"), index=False)

    # Friendly stdout summary
    print(f"\n[done] Wrote significance summary with {len(summary_df)} rows")
    print(summary_df.head(10).to_string(index=False))
    print("\nFriedman p-values for hamming_accuracy (BinaryVector):")
    for (metric, noise), p in sorted(focus_friedman.items()):
        print(f"  {metric:20s} noise={noise}  p = {p:.4g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/final_0624_summary")
    ap.add_argument("--output_dir", default="results/final_0624_summary/stats")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    run(args.results_dir, args.output_dir, alpha=args.alpha)


if __name__ == "__main__":
    main()
