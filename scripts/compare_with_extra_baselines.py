#!/usr/bin/env python3
"""Compare paper classifiers (PA-*, PR-*, BR, CC, CLR) vs the 3 extra rerun baselines
(ECC, LP, ML-kNN) using the rerun summary CSVs.

Aggregates mean across 8 datasets, per (classifier, metric, alpha).
"""
import csv
import re
from pathlib import Path

ROOT = Path("/Users/xuantruong/Documents/WORK/RESEARCH/preorders4MLC")
SUMMARY_DIR = ROOT / "results" / "final_20260514_v2_summary"

DATASETS = ["GpositivePseAAC", "VirusPseAAC", "emotions", "CHD_49",
            "scene", "PlantPseAAC", "Water-quality", "HumanPseAAC"]

ALGO_PRETTY = {
    "PartialOrder__Hamming__2": "PA-H-2",
    "PartialOrder__Hamming__None": "PA-H",
    "PartialOrder__Subset__2": "PA-S-2",
    "PartialOrder__Subset__None": "PA-S",
    "PreOrder__Hamming__2": "PR-H-2",
    "PreOrder__Hamming__None": "PR-H",
    "PreOrder__Subset__2": "PR-S-2",
    "PreOrder__Subset__None": "PR-S",
    "br": "BR", "cc": "CC", "clr": "CLR",
    "ecc": "ECC", "lp": "LP", "mlknn": "ML-kNN",
}

# (metric_prefix, direction, pretty)
BV_METRICS = [
    ("f1", "up", "F1"),
    ("hamming_accuracy", "up", "Hamming"),
    ("subset0_1", "up", "Subset"),
    ("afrd", "down", "AFRD"),
    ("mfrd", "down", "MFRD"),
]
ALPHAS = ["0.0", "0.1", "0.2", "0.3"]


def parse_mean(s):
    m = re.match(r"\s*([-+]?\d*\.?\d+)", s or "")
    return float(m.group(1)) if m else None


def load(dataset):
    path = SUMMARY_DIR / f"{dataset}_BinaryVector_summary.csv"
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            algo = row["Algorithm"]
            for col, cell in row.items():
                if col == "Algorithm" or "__" not in col:
                    continue
                metric, alpha = col.rsplit("__", 1)
                v = parse_mean(cell)
                if v is not None:
                    out[(algo, metric, alpha)] = v * 100.0  # to percent
    return out


def main():
    all_data = {ds: load(ds) for ds in DATASETS}

    # Aggregate across datasets: mean per (algo, metric, alpha)
    classifiers = list(ALGO_PRETTY.keys())

    rows = []
    for algo in classifiers:
        for metric_key, direction, metric_pretty in BV_METRICS:
            for a in ALPHAS:
                vals = [all_data[ds].get((algo, metric_key, a)) for ds in DATASETS]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                rows.append({
                    "classifier": ALGO_PRETTY[algo],
                    "metric": metric_pretty,
                    "direction": direction,
                    "alpha": a,
                    "mean": sum(vals) / len(vals),
                    "n": len(vals),
                })

    # Print summary tables per metric × alpha
    print("=" * 96)
    print("MEAN ACROSS 8 DATASETS, per (metric, α) — values in %")
    print("Arrow shows desired direction. Bold-equivalent = best (↑ for up, ↓ for down).")
    print("=" * 96)

    classifier_order = ["PA-H-2", "PA-H", "PA-S-2", "PA-S",
                        "PR-H-2", "PR-H", "PR-S-2", "PR-S",
                        "BR", "CC", "CLR",
                        "ECC", "LP", "ML-kNN"]

    for metric_pretty, direction in [(m[2], m[1]) for m in BV_METRICS]:
        arrow = "↑" if direction == "up" else "↓"
        print(f"\n--- {metric_pretty} ({arrow}) ---")
        # Table: rows = classifier, cols = α
        header = f"{'Classifier':<10} " + "  ".join(f"α={a:<4}" for a in ALPHAS)
        print(header)
        print("-" * len(header))
        # collect for ranking
        per_alpha = {a: {} for a in ALPHAS}
        for r in rows:
            if r["metric"] == metric_pretty:
                per_alpha[r["alpha"]][r["classifier"]] = r["mean"]
        # find best per alpha
        best = {}
        for a in ALPHAS:
            if direction == "up":
                best[a] = max(per_alpha[a].values())
            else:
                best[a] = min(per_alpha[a].values())
        for clf in classifier_order:
            row = [f"{clf:<10}"]
            for a in ALPHAS:
                v = per_alpha[a].get(clf)
                if v is None:
                    row.append("  -   ")
                else:
                    mark = "*" if abs(v - best[a]) < 1e-6 else " "
                    row.append(f"{v:6.2f}{mark}")
            print(" " + "  ".join(row[0:1]) + "  " + "  ".join(row[1:]))

    # ---- Head-to-head: PA/PR vs each extra baseline ----
    print()
    print("=" * 96)
    print("HEAD-TO-HEAD: BEST PA/PR vs each baseline (mean across datasets+α)")
    print("=" * 96)

    pa_pr = ["PA-H-2", "PA-H", "PA-S-2", "PA-S", "PR-H-2", "PR-H", "PR-S-2", "PR-S"]
    baselines = ["BR", "CC", "CLR", "ECC", "LP", "ML-kNN"]

    for metric_pretty, direction in [(m[2], m[1]) for m in BV_METRICS]:
        arrow = "↑" if direction == "up" else "↓"
        # For each classifier, average over all alphas
        flat = {}
        for r in rows:
            if r["metric"] == metric_pretty:
                flat.setdefault(r["classifier"], []).append(r["mean"])
        flat = {k: sum(v)/len(v) for k, v in flat.items()}
        if direction == "up":
            best_pa_pr = max(flat[c] for c in pa_pr)
            best_clf = max(pa_pr, key=lambda c: flat[c])
        else:
            best_pa_pr = min(flat[c] for c in pa_pr)
            best_clf = min(pa_pr, key=lambda c: flat[c])
        print(f"\n{metric_pretty} ({arrow})    best PA/PR = {best_clf} ({best_pa_pr:.2f})")
        for b in baselines:
            v = flat.get(b)
            if v is None:
                continue
            diff = v - best_pa_pr
            sign = "+" if diff >= 0 else ""
            if direction == "up":
                winner = "baseline" if v > best_pa_pr else "PA/PR"
            else:
                winner = "baseline" if v < best_pa_pr else "PA/PR"
            print(f"  {b:<8} = {v:6.2f}   ({sign}{diff:+.2f} vs best PA/PR)   → winner: {winner}")

    # Per-dataset overview for the 3 new baselines on F1
    print()
    print("=" * 96)
    print("PER-DATASET F1 (↑): best of {PA/PR} vs each new baseline, averaged over α")
    print("=" * 96)
    for ds in DATASETS:
        d = all_data[ds]
        # compute mean over α per algo, F1
        per = {}
        for algo, pretty in ALGO_PRETTY.items():
            vals = [d.get((algo, "f1", a)) for a in ALPHAS]
            vals = [v for v in vals if v is not None]
            if vals:
                per[pretty] = sum(vals)/len(vals)
        if not per:
            continue
        best_papr = max(per[c] for c in pa_pr if c in per)
        new_b = {b: per.get(b, None) for b in ["ECC", "LP", "ML-kNN"]}
        wins = sum(1 for v in new_b.values() if v is not None and v > best_papr)
        print(f"  {ds:<18} bestPA/PR={best_papr:5.2f}  ECC={new_b['ECC'] or 0:5.2f}  LP={new_b['LP'] or 0:5.2f}  ML-kNN={new_b['ML-kNN'] or 0:5.2f}   (baselines beating bestPA/PR: {wins}/3)")


if __name__ == "__main__":
    main()
