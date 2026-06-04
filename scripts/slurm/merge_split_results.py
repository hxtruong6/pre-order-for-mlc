"""Merge per-(repeat, fold) pickle splits into the consolidated form.

Each split sbatch job writes a partial pickle named
``dataset_<name>_noisy_<r>[_clr|_br|_cc]_r<R>_f<F>.pkl`` containing the
records for one (repeat, fold) cell. This script globs those partials
for a single (dataset, noise) cell and concatenates them back into the
flat ``dataset_<name>_noisy_<r>[_clr|_br|_cc].pkl`` that
:mod:`scripts.evaluate` understands.

Usage:
    python scripts/ablations/merge_split_results.py \\
        --dataset medical --noise_rate 0.0 --results_dir results/full_medical

The merger overwrites any existing consolidated pickle.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys

_SUFFIXES = [
    "", "_clr", "_br", "_cc",
    "_mlknn", "_ecc", "_lp",
    # LightGBM-trained extras use the bl_suffix from train_extra_baselines.py
    # (`bl_suffix = f"_{base_learner}" if base_learner != "rf" else ""`).
    "_mlknn_lgbm", "_ecc_lgbm", "_lp_lgbm",
]


def _merge_one(results_dir: str, dataset_name: str, noisy_rate: float, suffix: str) -> int:
    base = f"dataset_{dataset_name.lower()}_noisy_{noisy_rate}{suffix}"
    pattern = os.path.join(results_dir, f"{base}_r*_f*.pkl")
    parts = sorted(glob.glob(pattern))
    if not parts:
        return 0

    # Sort by (repeat, fold) so the merged list is deterministic.
    rx = re.compile(rf"{re.escape(base)}_r(\d+)_f(\d+)\.pkl$")

    def _key(path: str) -> tuple[int, int]:
        m = rx.search(path)
        if not m:
            return (10**9, 10**9)
        return (int(m.group(1)), int(m.group(2)))

    parts.sort(key=_key)

    merged: list = []
    for path in parts:
        with open(path, "rb") as f:
            merged.extend(pickle.load(f))

    out_path = os.path.join(results_dir, f"{base}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(merged, f)
    print(f"[merge] {out_path} ← {len(parts)} parts, {len(merged)} records")
    return len(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--noise_rate", type=float, required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    total = 0
    for suf in _SUFFIXES:
        total += _merge_one(args.results_dir, args.dataset, args.noise_rate, suf)

    if total == 0:
        print(
            f"[merge] no split partials found under {args.results_dir} "
            f"for {args.dataset} noisy={args.noise_rate}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
