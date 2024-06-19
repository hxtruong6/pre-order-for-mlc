# Define constant variables to re-use it

from enum import Enum


class BaseLearnerName(Enum):
    RF = "RF"
    ETC = "ETC"
    XGBoost = "XGBoost"
    LightGBM = "LightGBM"


RANDOM_STATE = 6


class TargetMetric(Enum):
    Hamming = "Hamming"
    Subset = "Subset"
