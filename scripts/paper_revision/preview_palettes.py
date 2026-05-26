#!/usr/bin/env python3
"""Render a single PDF previewing color-palette options for the paper revision.

Generates a multi-page PDF showing:
 - current Original (with cyan) vs Original (cyan -> teal) on a paired-bars mock
 - 3 candidate Enhanced palettes (Okabe-Ito, Tol bright, ColorBrewer Dark2)
 - each palette is also shown as it would look in grayscale (B&W print)
 - a colorblind-simulation row (deuteranopia approximation by channel mixing)

Output: paper_revision/palette_preview.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PDF = REPO_ROOT / "paper_revision" / "palette_preview.pdf"


# ---- palettes -------------------------------------------------------------

ORIGINAL_CURRENT = {
    "name": "Original (current, with cyan)",
    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#00FFFF"],
    "labels": ["blue", "orange", "green", "red", "cyan"],
}
ORIGINAL_FIXED = {
    "name": "Original (cyan -> Tol teal #117733)",
    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#117733"],
    "labels": ["blue", "orange", "green", "red", "teal"],
}
ORIGINAL_BLACK = {
    "name": "Original (cyan -> black #000000)",
    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#000000"],
    "labels": ["blue", "orange", "green", "red", "black"],
}

# Sequential palettes for the NOISE-LEVEL dimension (α=0.0, 0.1, 0.2, 0.3).
# Noise is ordinal so qualitative palettes are wrong here — we want a
# monotone luminance ramp that survives both grayscale and CB.
NOISE_SEQUENTIAL_OPTIONS = [
    {
        "name": "Noise Option A: viridis (matplotlib default, CB-safe)",
        "cmap": "viridis",
    },
    {
        "name": "Noise Option B: cividis (CB-optimized, prints well)",
        "cmap": "cividis",
    },
    {
        "name": "Noise Option C: ColorBrewer YlOrRd (warm sequential)",
        "cmap": "YlOrRd",
    },
    {
        "name": "Noise Option D: ColorBrewer Blues (cool sequential)",
        "cmap": "Blues",
    },
]


ENHANCED_OPTIONS = [
    {
        "name": "Option 1: Okabe-Ito (CB gold standard)",
        "colors": [
            "#000000", "#E69F00", "#56B4E9", "#009E73",
            "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
        ],
        "labels": [
            "black", "orange", "skyblue", "bluegreen",
            "yellow", "blue", "vermillion", "purple",
        ],
    },
    {
        "name": "Option 2: Paul Tol bright",
        "colors": [
            "#4477AA", "#EE6677", "#228833", "#CCBB44",
            "#66CCEE", "#AA3377", "#BBBBBB",
        ],
        "labels": ["blue", "red", "green", "yellow", "cyan", "purple", "grey"],
    },
    {
        "name": "Option 3: ColorBrewer Dark2",
        "colors": [
            "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
            "#66A61E", "#E6AB02", "#A6761D", "#666666",
        ],
        "labels": [
            "teal", "orange", "purple", "magenta",
            "olive", "gold", "brown", "grey",
        ],
    },
]


# ---- helpers --------------------------------------------------------------

def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance."""
    y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return np.array([y, y, y])


def simulate_deuteranopia(rgb: np.ndarray) -> np.ndarray:
    """Approximate deuteranopia (red-green CB) via Brettel-style matrix."""
    m = np.array([
        [0.625, 0.375, 0.0],
        [0.700, 0.300, 0.0],
        [0.0, 0.300, 0.700],
    ])
    out = m @ rgb
    return np.clip(out, 0, 1)


# ---- swatch + chart panels -----------------------------------------------

def draw_swatches(ax, colors, labels, title, transform=None):
    ax.set_title(title, fontsize=9, loc="left")
    n = len(colors)
    for i, (c, lab) in enumerate(zip(colors, labels)):
        rgb = hex_to_rgb(c)
        if transform is not None:
            rgb = transform(rgb)
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=rgb))
        ax.text(
            i + 0.45, -0.25, lab,
            ha="center", va="top", fontsize=7,
        )
    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(-0.7, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_paired_bars_mock(ax, colors, title):
    """Bars resembling paired_bars (Δ across noise) using the first 5 colors."""
    rng = np.random.default_rng(0)
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    methods = ["BR", "MLkNN", "CC", "Stacked", "Ours"]
    width = 0.16
    x = np.arange(len(noise_levels))
    for i, m in enumerate(methods):
        vals = rng.uniform(-2, 6, len(noise_levels)) - i * 0.3
        ax.bar(
            x + (i - 2) * width, vals, width,
            color=colors[i % len(colors)], label=m,
            edgecolor="black", linewidth=0.3,
        )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"α={a}" for a in noise_levels], fontsize=7)
    ax.set_ylabel("Δ F1 (pp)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, ncol=5, loc="upper right", frameon=False)
    ax.tick_params(labelsize=7)


def draw_lines_mock(ax, colors, title):
    x = np.linspace(0, 1, 25)
    rng = np.random.default_rng(1)
    for i in range(min(7, len(colors))):
        y = np.sin(2 * np.pi * x + i * 0.6) * 0.4 + i * 0.15 + rng.normal(
            0, 0.02, len(x)
        )
        ax.plot(x, y, color=colors[i], linewidth=1.6, label=f"s{i + 1}")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, ncol=4, frameon=False, loc="lower right")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("noise α", fontsize=8)
    ax.set_ylabel("metric", fontsize=8)


# ---- page builders --------------------------------------------------------

def render_palette_page(pdf, palette, kind="enhanced"):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(palette["name"], fontsize=13, y=0.97)

    # Row 1: swatches (color / grayscale / deuteranopia)
    gs = fig.add_gridspec(
        4, 3, height_ratios=[0.7, 0.7, 0.7, 2.3], hspace=0.55, wspace=0.25,
        left=0.05, right=0.97, top=0.92, bottom=0.06,
    )

    ax0 = fig.add_subplot(gs[0, :])
    draw_swatches(ax0, palette["colors"], palette["labels"], "Color")

    ax1 = fig.add_subplot(gs[1, :])
    draw_swatches(
        ax1, palette["colors"], palette["labels"],
        "Grayscale (B&W print)", transform=to_grayscale,
    )

    ax2 = fig.add_subplot(gs[2, :])
    draw_swatches(
        ax2, palette["colors"], palette["labels"],
        "Deuteranopia simulation", transform=simulate_deuteranopia,
    )

    # Row 2: 3 mock charts: paired bars, paired bars grayscale, lines
    axA = fig.add_subplot(gs[3, 0])
    draw_paired_bars_mock(axA, palette["colors"], "Mock paired_bars (color)")

    axB = fig.add_subplot(gs[3, 1])
    gray_colors = [to_grayscale(hex_to_rgb(c)) for c in palette["colors"]]
    draw_paired_bars_mock(axB, gray_colors, "Mock paired_bars (B&W)")

    axC = fig.add_subplot(gs[3, 2])
    draw_lines_mock(axC, palette["colors"], "Mock per-noise curves")

    pdf.savefig(fig)
    plt.close(fig)


def _sample_cmap(name: str, n: int = 4) -> list[np.ndarray]:
    cmap = plt.get_cmap(name)
    # Avoid the extreme ends (too light/dark for legibility).
    xs = np.linspace(0.15, 0.85, n)
    return [np.array(cmap(x)[:3]) for x in xs]


def draw_noise_bars_mock(ax, colors, title):
    """Mock: bars where COLOR encodes noise level (4 levels)."""
    methods = ["BR", "MLkNN", "CC", "Stacked", "Ours"]
    rng = np.random.default_rng(2)
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    width = 0.18
    x = np.arange(len(methods))
    for j, a in enumerate(noise_levels):
        vals = rng.uniform(0.45, 0.80, len(methods)) - j * 0.04
        ax.bar(
            x + (j - 1.5) * width, vals, width,
            color=colors[j], label=f"α={a}",
            edgecolor="black", linewidth=0.3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel("F1", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, ncol=4, loc="upper right", frameon=False)
    ax.tick_params(labelsize=7)


def render_noise_page(pdf, opt):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(opt["name"], fontsize=13, y=0.97)

    colors = _sample_cmap(opt["cmap"], n=4)
    labels = ["α=0.0", "α=0.1", "α=0.2", "α=0.3"]
    hex_colors = [
        "#{:02X}{:02X}{:02X}".format(*[int(round(v * 255)) for v in c])
        for c in colors
    ]

    gs = fig.add_gridspec(
        4, 3, height_ratios=[0.7, 0.7, 0.7, 2.3], hspace=0.55, wspace=0.25,
        left=0.05, right=0.97, top=0.92, bottom=0.06,
    )
    palette = {"colors": hex_colors, "labels": labels}

    ax0 = fig.add_subplot(gs[0, :])
    draw_swatches(ax0, palette["colors"], palette["labels"],
                  "Color (noise α: light=low, dark=high)")

    ax1 = fig.add_subplot(gs[1, :])
    draw_swatches(ax1, palette["colors"], palette["labels"],
                  "Grayscale (B&W print)", transform=to_grayscale)

    ax2 = fig.add_subplot(gs[2, :])
    draw_swatches(ax2, palette["colors"], palette["labels"],
                  "Deuteranopia simulation", transform=simulate_deuteranopia)

    axA = fig.add_subplot(gs[3, 0])
    draw_noise_bars_mock(axA, colors, "Mock: color = noise α (color)")

    axB = fig.add_subplot(gs[3, 1])
    gray_colors = [to_grayscale(c) for c in colors]
    draw_noise_bars_mock(axB, gray_colors, "Mock: color = noise α (B&W)")

    axC = fig.add_subplot(gs[3, 2])
    x = np.linspace(0, 1, 25)
    rng = np.random.default_rng(3)
    for i in range(4):
        y = 0.7 - i * 0.08 + 0.04 * np.sin(2 * np.pi * x + i) + rng.normal(
            0, 0.01, len(x)
        )
        axC.plot(x, y, color=colors[i], linewidth=1.8, label=labels[i])
    axC.set_title("Mock: per-α curve", fontsize=9)
    axC.set_xlabel("position", fontsize=8)
    axC.set_ylabel("F1", fontsize=8)
    axC.legend(fontsize=6, frameon=False, loc="lower right")
    axC.tick_params(labelsize=7)

    pdf.savefig(fig)
    plt.close(fig)


def render_cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.78, "Palette preview", ha="center", fontsize=22,
             weight="bold")
    fig.text(0.5, 0.72, "for paper revision (original + enhanced)",
             ha="center", fontsize=12)
    lines = [
        "Pages — QUALITATIVE (method / dataset dimension):",
        "  1. Original (current, with cyan) — baseline",
        "  2. Original (cyan -> Tol teal #117733) — proposed fix",
        "  3. Enhanced Option 1 — Okabe-Ito (CB gold standard)",
        "  4. Enhanced Option 2 — Paul Tol bright",
        "  5. Enhanced Option 3 — ColorBrewer Dark2",
        "",
        "Pages — SEQUENTIAL (noise level α, ordinal):",
        "  6. Noise A — viridis",
        "  7. Noise B — cividis",
        "  8. Noise C — YlOrRd (warm)",
        "  9. Noise D — Blues (cool)",
        "",
        "Each page shows:",
        "  - color swatches",
        "  - grayscale rendering (simulates B&W print)",
        "  - deuteranopia simulation (red-green CB)",
        "  - mock charts (color + B&W) + line plot",
    ]
    fig.text(0.12, 0.55, "\n".join(lines), fontsize=11, family="monospace",
             va="top")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        render_cover(pdf)
        render_palette_page(pdf, ORIGINAL_CURRENT, kind="original")
        render_palette_page(pdf, ORIGINAL_FIXED, kind="original")
        render_palette_page(pdf, ORIGINAL_BLACK, kind="original")
        for opt in ENHANCED_OPTIONS:
            render_palette_page(pdf, opt, kind="enhanced")
        for opt in NOISE_SEQUENTIAL_OPTIONS:
            render_noise_page(pdf, opt)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
