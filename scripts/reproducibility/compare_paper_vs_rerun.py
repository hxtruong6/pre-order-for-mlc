#!/usr/bin/env python3
"""Compare hardcoded paper.tex values vs the rerun in results/final_20260514_v2_summary/.

Outputs a unified diff CSV plus a stdout summary of the biggest deltas.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY_DIR = ROOT / "results" / "final_20260514_v2_summary"
OUT_DIR = ROOT / "results" / "comparison_paper_vs_rerun"

PAPER_TABLES = [
    ("T2_main_bv", 720, 1779, "BinaryVector"),
    ("T3_main_pa", 1786, 2624, "PartialAbstention"),
    ("T4_app_bv1", 3051, 3765, "BinaryVector"),
    ("T5_app_bv2", 3767, 4831, "BinaryVector"),
    ("T6_app_pa1", 4840, 5404, "PartialAbstention"),
    ("T7_app_pa2", 5407, 6244, "PartialAbstention"),
]

DATASET_MAP = {
    "GpositivePse": "GpositivePseAAC",
    "plantPse": "PlantPseAAC",
    "HumanPse": "HumanPseAAC",
    "VirusPse": "VirusPseAAC",
    "CHD-49": "CHD_49",
    "Emotion": "emotions",
    "Scene": "scene",
    "Water-quality": "Water-quality",
}

CLASSIFIER_MAP = {
    "PA-H-2": "PartialOrder__Hamming__2",
    "PA-H": "PartialOrder__Hamming__None",
    "PA-S-2": "PartialOrder__Subset__2",
    "PA-S": "PartialOrder__Subset__None",
    "PR-H-2": "PreOrder__Hamming__2",
    "PR-H": "PreOrder__Hamming__None",
    "PR-S-2": "PreOrder__Subset__2",
    "PR-S": "PreOrder__Subset__None",
    "BR": "br",
    "CC": "cc",
    "CLR": "clr",
}

# Paper metric label -> column prefix in summary CSV (before "__<alpha>")
METRIC_MAP_BV = {
    "F1": "f1",
    "Hamming": "hamming_accuracy",
    "Subset": "subset0_1",
    "AFRD": "afrd",
    "MFRD": "mfrd",
}
METRIC_MAP_PA = {
    "F1": "f1_pa",
    "Hamming": "hamming_accuracy_pa",
    "Subset": "subset0_1_pa",
    "AABS": "aabs",
    "ABS": "abs",
}


def load_paper_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_mean(cell):
    """Parse a rerun cell like '0.6629±0.0354' → 0.6629."""
    if cell is None or cell == "":
        return None
    m = re.match(r"\s*([-+]?\d*\.?\d+)", cell)
    return float(m.group(1)) if m else None


def load_rerun(dataset_file, kind):
    """Load rerun summary into dict: (algorithm, metric_prefix, alpha_str) -> mean value."""
    path = SUMMARY_DIR / f"{dataset_file}_{kind}_summary.csv"
    if not path.exists():
        return None
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row["Algorithm"]
            for col, cell in row.items():
                if col == "Algorithm":
                    continue
                if "__" not in col:
                    continue
                metric_prefix, alpha_str = col.rsplit("__", 1)
                v = parse_mean(cell)
                if v is None:
                    continue
                out[(algo, metric_prefix, alpha_str)] = v
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_rows = []
    rerun_cache = {}

    for tname, start, end, kind in PAPER_TABLES:
        csv_path = OUT_DIR / f"{tname}_paper.csv"
        # Run extractor
        import subprocess
        subprocess.run(
            ["python", str(ROOT / "scripts" / "reproducibility" / "extract_paper_tables.py"),
             str(start), str(end), str(csv_path)],
            check=True,
        )
        paper_rows = load_paper_csv(csv_path)
        metric_map = METRIC_MAP_BV if kind == "BinaryVector" else METRIC_MAP_PA

        for r in paper_rows:
            ds_paper = r["dataset"]
            ds_file = DATASET_MAP.get(ds_paper)
            if ds_file is None:
                out_rows.append({
                    "table": tname, "dataset_paper": ds_paper, "metric_paper": r["metric"],
                    "classifier_paper": r["classifier"], "alpha": r["alpha"],
                    "paper_value_pct": r["value"], "rerun_value_pct": "",
                    "delta_pct": "", "abs_delta_pct": "", "note": "UNKNOWN_DATASET",
                })
                continue
            metric_prefix = metric_map.get(r["metric"])
            if metric_prefix is None:
                out_rows.append({
                    "table": tname, "dataset_paper": ds_paper, "metric_paper": r["metric"],
                    "classifier_paper": r["classifier"], "alpha": r["alpha"],
                    "paper_value_pct": r["value"], "rerun_value_pct": "",
                    "delta_pct": "", "abs_delta_pct": "", "note": "UNKNOWN_METRIC",
                })
                continue
            algo = CLASSIFIER_MAP.get(r["classifier"])
            if algo is None:
                out_rows.append({
                    "table": tname, "dataset_paper": ds_paper, "metric_paper": r["metric"],
                    "classifier_paper": r["classifier"], "alpha": r["alpha"],
                    "paper_value_pct": r["value"], "rerun_value_pct": "",
                    "delta_pct": "", "abs_delta_pct": "", "note": "UNKNOWN_CLASSIFIER",
                })
                continue
            # Load rerun
            cache_key = (ds_file, kind)
            if cache_key not in rerun_cache:
                rerun_cache[cache_key] = load_rerun(ds_file, kind)
            rerun = rerun_cache[cache_key]
            if rerun is None:
                out_rows.append({
                    "table": tname, "dataset_paper": ds_paper, "metric_paper": r["metric"],
                    "classifier_paper": r["classifier"], "alpha": r["alpha"],
                    "paper_value_pct": r["value"], "rerun_value_pct": "",
                    "delta_pct": "", "abs_delta_pct": "", "note": "RERUN_FILE_MISSING",
                })
                continue
            # alpha in paper csv is "0.0".."0.3" but stored as float string "0.0"; rerun keys use "0.0"
            alpha_str = r["alpha"]
            # Rerun key alpha is like "0.0"; we want to match
            rerun_v = rerun.get((algo, metric_prefix, alpha_str))
            if rerun_v is None:
                out_rows.append({
                    "table": tname, "dataset_paper": ds_paper, "metric_paper": r["metric"],
                    "classifier_paper": r["classifier"], "alpha": r["alpha"],
                    "paper_value_pct": r["value"], "rerun_value_pct": "",
                    "delta_pct": "", "abs_delta_pct": "", "note": "RERUN_MISSING_KEY",
                })
                continue
            paper_v = float(r["value"])
            rerun_pct = rerun_v * 100.0
            delta = rerun_pct - paper_v
            out_rows.append({
                "table": tname, "dataset_paper": ds_paper, "metric_paper": r["metric"],
                "classifier_paper": r["classifier"], "alpha": r["alpha"],
                "paper_value_pct": f"{paper_v:.4f}",
                "rerun_value_pct": f"{rerun_pct:.4f}",
                "delta_pct": f"{delta:+.4f}",
                "abs_delta_pct": f"{abs(delta):.4f}",
                "note": "",
            })

    # Write unified CSV
    out_csv = OUT_DIR / "all_diffs.csv"
    fieldnames = ["table", "dataset_paper", "metric_paper", "classifier_paper", "alpha",
                  "paper_value_pct", "rerun_value_pct", "delta_pct", "abs_delta_pct", "note"]
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} comparison rows → {out_csv}")

    # ---- Reporting ----
    total = len(out_rows)
    matched = [r for r in out_rows if r["note"] == ""]
    notes_count = {}
    for r in out_rows:
        if r["note"]:
            notes_count[r["note"]] = notes_count.get(r["note"], 0) + 1
    print(f"\nTotal rows: {total}, matched: {len(matched)}, with notes: {total - len(matched)}")
    for k, v in sorted(notes_count.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # Bucketize deltas
    buckets = {"<=0.5": 0, "<=1.0": 0, "<=2.0": 0, "<=5.0": 0, ">5.0": 0}
    for r in matched:
        d = float(r["abs_delta_pct"])
        if d <= 0.5: buckets["<=0.5"] += 1
        elif d <= 1.0: buckets["<=1.0"] += 1
        elif d <= 2.0: buckets["<=2.0"] += 1
        elif d <= 5.0: buckets["<=5.0"] += 1
        else: buckets[">5.0"] += 1
    print("\nAbsolute delta distribution (pp = percentage points):")
    for k, v in buckets.items():
        pct = (v / len(matched) * 100) if matched else 0
        print(f"  {k:>7} pp: {v:5d}  ({pct:5.1f}%)")

    # Top 30 biggest deltas
    matched_sorted = sorted(matched, key=lambda r: -float(r["abs_delta_pct"]))
    print("\nTop 30 largest abs deltas (rerun − paper, in pp):")
    print(f"  {'table':<12} {'dataset':<14} {'metric':<8} {'clf':<8} α    paper   rerun   Δ")
    for r in matched_sorted[:30]:
        print(f"  {r['table']:<12} {r['dataset_paper']:<14} {r['metric_paper']:<8} {r['classifier_paper']:<8} {r['alpha']:<4} {r['paper_value_pct']:>7} {r['rerun_value_pct']:>7} {r['delta_pct']:>8}")

    # Per-classifier mean abs delta
    print("\nMean |Δ| (pp) by classifier:")
    by_clf = {}
    for r in matched:
        by_clf.setdefault(r["classifier_paper"], []).append(float(r["abs_delta_pct"]))
    for clf, vals in sorted(by_clf.items()):
        print(f"  {clf:<8} n={len(vals):4d}  mean={sum(vals)/len(vals):.3f}  max={max(vals):.3f}")

    # Per-metric mean abs delta
    print("\nMean |Δ| (pp) by metric:")
    by_m = {}
    for r in matched:
        by_m.setdefault(r["metric_paper"], []).append(float(r["abs_delta_pct"]))
    for m, vals in sorted(by_m.items()):
        print(f"  {m:<8} n={len(vals):4d}  mean={sum(vals)/len(vals):.3f}  max={max(vals):.3f}")

    # Per-dataset
    print("\nMean |Δ| (pp) by dataset:")
    by_d = {}
    for r in matched:
        by_d.setdefault(r["dataset_paper"], []).append(float(r["abs_delta_pct"]))
    for d, vals in sorted(by_d.items()):
        print(f"  {d:<14} n={len(vals):4d}  mean={sum(vals)/len(vals):.3f}  max={max(vals):.3f}")


if __name__ == "__main__":
    main()
