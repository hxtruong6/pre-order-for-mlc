"""Shared constants and enum types.

Centralises the base learner names and target metric enums used across
:mod:`training_orchestrator`, :mod:`inference_models`, and
:mod:`searching_algorithms`, along with the canonical random seed that
controls every reproducible split and noise draw in the paper.
"""

from enum import Enum


class BaseLearnerName(Enum):
    """Identifier for the supported base learners."""

    RF = "RF"
    ETC = "ETC"
    XGBoost = "XGBoost"
    LightGBM = "LightGBM"


# Used by every split, fold, and Bernoulli noise draw. Do not change
# casually -- result directories in `results/` are tied to this seed.
RANDOM_STATE = 6


class TargetMetric(Enum):
    """Loss the ILP search optimises for in :mod:`searching_algorithms`."""

    Hamming = "Hamming"
    Subset = "Subset"
