#!/usr/bin/env python3
"""Compute per-dataset statistics (N, P, K, MeanIR, CVIR) and cache to JSON.

These values feed the datasets table emitted by build_tex.py. Cached so
that re-running build_tex.py doesn't reload the ARFF/NPY files every time.

Usage::

    python scripts/paper_revision/compute_dataset_stats.py
    # writes paper_revision/dataset_stats.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUT_JSON = REPO_ROOT / "paper_revision" / "dataset_stats.json"

# (key, display_name, source_file, K)
DATASETS = [
    # 6 original-paper datasets. COMETA convention places labels at the END of
    # the ARFF for most of these (Yeast is reordered with labels at the START).
    ("GpositivePseAAC", "GpositivePseAAC", "GpositivePseAAC.arff", 4),
    ("emotions", "Emotions", "emotions.arff", 6),
    ("scene", "Scene", "scene.arff", 6),
    ("PlantPseAAC", "PlantPseAAC", "PlantPseAAC.arff", 12),
    ("HumanPseAAC", "HumanPseAAC", "HumanPseAAC.arff", 14),
    ("Yeast", "Yeast", "Yeast.arff", 14),
    # 3 revision-extension datasets.
    ("CHD_49", "CHD_49", "CHD_49.arff", 6),
    ("Water-quality", "Water-quality", "Water-quality.arff", 14),
    ("enron", "enron", "enron.arff", 53),
]

# Datasets where labels live at the END of the ARFF (COMETA convention).
# Yeast / CHD_49 / Water-quality have been reordered with labels at the START
# (see the @relation header) so they are NOT in this set.
TARGET_IN_END = {
    "birds.arff",
    "GpositivePseAAC.arff",
    "VirusPseAAC.arff",
    "emotions.arff",
    "scene.arff",
    "PlantPseAAC.arff",
    "HumanPseAAC.arff",
}


def load_xy(source: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / source
    if source.endswith(".npy"):
        x = np.load(path)
        y = np.load(str(path).replace("_features.npy", "_labels.npy"))
        y = np.where(y < 0, 0, y).astype(int)
        return x, y
    try:
        data, _ = arff.loadarff(str(path))
        df = pd.DataFrame(data)
    except (ValueError, NotImplementedError):
        # Sparse ARFF — fall back to liac-arff densification.
        import arff as liac

        with open(path) as f:
            obj = liac.load(f)
        names = [a[0] for a in obj["attributes"]]
        rows = obj["data"]
        # liac returns sparse rows as dicts; densify with zeros.
        full = np.zeros((len(rows), len(names)), dtype=float)
        for i, r in enumerate(rows):
            if isinstance(r, dict):
                for j, v in r.items():
                    full[i, j] = float(v)
            else:
                for j, v in enumerate(r):
                    full[i, j] = float(v) if v is not None else 0.0
        df = pd.DataFrame(full, columns=names)

    if source in TARGET_IN_END:
        x_df, y_df = df.iloc[:, :-k], df.iloc[:, -k:]
    else:
        x_df, y_df = df.iloc[:, k:], df.iloc[:, :k]

    # scipy.io.arff returns nominal labels as bytes (b'0' / b'1'); decode
    # before casting to int so the comparison + astype don't truncate to 0.
    def _to_int01(col: pd.Series) -> np.ndarray:
        if col.dtype == object:
            return col.map(
                lambda v: int(v.decode() if isinstance(v, bytes) else v)
            ).to_numpy()
        return col.to_numpy().astype(int)

    y = np.column_stack([_to_int01(y_df.iloc[:, j]) for j in range(k)])
    y = np.where(y < 0, 0, y).astype(int)
    return x_df.to_numpy(), y


def compute_imbalance(y: np.ndarray) -> tuple[float, float]:
    """Return (MeanIR, CVIR) following Charte et al. 2015.

    For each label k, IR_k = max_freq / freq_k (skipping K=0 labels).
    MeanIR = mean(IR_k); CVIR = std(IR_k) / MeanIR.
    """
    positives = y.sum(axis=0).astype(float)
    positives = positives[positives > 0]
    max_freq = positives.max()
    ir = max_freq / positives
    mean_ir = float(ir.mean())
    cvir = float(ir.std(ddof=0) / mean_ir) if mean_ir > 0 else 0.0
    return mean_ir, cvir


def main() -> None:
    stats: dict[str, dict] = {}
    for key, display, source, k in DATASETS:
        try:
            x, y = load_xy(source, k)
        except FileNotFoundError as e:
            print(f"  ! {key}: missing data file ({e}); skipping")
            continue
        n, p = x.shape
        mean_ir, cvir = compute_imbalance(y)
        stats[key] = {
            "display": display,
            "N": int(n),
            "P": int(p),
            "K": int(k),
            "MeanIR": round(mean_ir, 2),
            "CVIR": round(cvir, 2),
            "source": "computed (full dataset)",
        }
        print(
            f"  {key:25s}  N={n:>6d}  P={p:>5d}  K={k:>3d}  "
            f"MeanIR={mean_ir:6.2f}  CVIR={cvir:5.2f}"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
