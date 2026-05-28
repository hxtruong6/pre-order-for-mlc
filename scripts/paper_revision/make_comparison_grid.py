#!/usr/bin/env python3
"""Assemble the 3-version comparison panels into a single labelled grid PNG.

Layout: rows = sample panels, columns = v1 / v2 / v3
Run:
    python scripts/paper_revision/make_comparison_grid.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "paper_revision" / "figures_compare"

VERSIONS = [
    ("v1_high",        "V1 — High impact\n(shape sizes + jitter + legend)"),
    ("v2_high_medium", "V2 — + Medium\n(black edge + loose dots + tight y)"),
    ("v3_all",         "V3 — + Low\n(faint grid + thin line)"),
]

SAMPLES = [
    ("emotions_PartialAbstention_aabs",         "emotions · PA · aabs"),
    ("GpositivePseAAC_PartialAbstention_aabs",  "GpositivePseAAC · PA · aabs"),
    ("CHD_49_PartialAbstention_aabs",           "CHD_49 · PA · aabs"),
    ("emotions_BinaryVector_hamming_accuracy",   "emotions · BV · hamming_acc"),
]

n_rows = len(SAMPLES)
n_cols = len(VERSIONS)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.2))
fig.patch.set_facecolor("#f8f8f8")

for col, (ver_dir, ver_label) in enumerate(VERSIONS):
    axes[0, col].set_title(ver_label, fontsize=10, fontweight="bold", pad=8,
                           color="#222222")

for row, (stem, row_label) in enumerate(SAMPLES):
    axes[row, 0].set_ylabel(row_label, fontsize=8, rotation=90,
                            labelpad=6, color="#444444")
    for col, (ver_dir, _) in enumerate(VERSIONS):
        ax = axes[row, col]
        img_path = BASE / ver_dir / f"{stem}.png"
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        ax.axis("off")
        # thin border to separate cells
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_edgecolor("#cccccc")

fig.suptitle("Panel styling comparison — same data, three progressive improvements",
             fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.5)

out = BASE / "COMPARISON_GRID.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {out}")
