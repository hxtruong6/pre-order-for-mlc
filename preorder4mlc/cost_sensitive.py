"""Cost-sensitive prediction helpers (paper Appendix G).

Decision-theoretic Bayes-optimal predictor for cost-sensitive Hamming:

    cost(ŷ=1 | y) = c_FP · (1 - P(y=1))
    cost(ŷ=0 | y) = c_FN · P(y=1)

so the optimal label is ŷ=1 iff P(y=1) > c_FP / (c_FP + c_FN).

This is intentionally classifier-agnostic: it operates on per-label marginal
probabilities Y_proba (shape (n_samples, n_labels)) regardless of how those
were estimated (BR, CC, PA/PR marginals via Pij, etc.).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cost_sensitive_threshold(cost_fp: float, cost_fn: float) -> float:
    """Return the Bayes-optimal marginal-probability threshold for cost-sensitive Hamming."""
    total = cost_fp + cost_fn
    if total <= 0:
        raise ValueError("cost_fp + cost_fn must be positive")
    return float(cost_fp / total)


def predict_cost_sensitive(
    Y_proba: NDArray,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> NDArray:
    """Binarise marginal probabilities at the cost-sensitive threshold.

    Args:
        Y_proba: array shape (n_samples, n_labels), values in [0, 1]
        cost_fp: cost of a false positive (predicting 1 when truth is 0)
        cost_fn: cost of a false negative (predicting 0 when truth is 1)

    Returns:
        Binary array shape (n_samples, n_labels). With c_FP = c_FN = 1 the
        threshold collapses to 0.5 (plain Hamming-optimal).
    """
    thr = cost_sensitive_threshold(cost_fp, cost_fn)
    Y_proba = np.asarray(Y_proba, dtype=float)
    return (Y_proba > thr).astype(int)


def pareto_sweep_costs(start: float = 0.5, stop: float = 5.0, n: int = 10) -> list[tuple[float, float]]:
    """Generate (cost_fp, cost_fn) pairs sweeping the F1↔Hamming Pareto curve.

    Holds c_FP = 1 and varies c_FN logarithmically. High c_FN ⇒ avoid FN ⇒ predict
    more positives ⇒ higher recall / F1, lower Hamming.
    """
    cs_fn = np.geomspace(start, stop, n)
    return [(1.0, float(c)) for c in cs_fn]
