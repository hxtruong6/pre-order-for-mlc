"""Uniform estimator interface over scikit-learn and LightGBM backends.

:class:`Estimator` is the single adapter every other module talks to;
its constructor accepts a :class:`constants.BaseLearnerName` and hides
the differences between :class:`sklearn.ensemble.RandomForestClassifier`,
:class:`sklearn.ensemble.ExtraTreesClassifier`,
:class:`sklearn.ensemble.GradientBoostingClassifier` (XGBoost in our
nomenclature), and :class:`lightgbm.LGBMClassifier`.
"""

import os
from logging import INFO, basicConfig

from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from preorder4mlc.constants import RANDOM_STATE, BaseLearnerName

basicConfig(level=INFO)  # type: ignore

number_of_cores: int = os.cpu_count() if os.cpu_count() is not None else 1  # type: ignore
# log(INFO, f"Number of cores: {number_of_cores}")

# Whether to wrap the base learner in CalibratedClassifierCV (isotonic).
# Opt-in: default OFF so existing pipelines / paper-equivalent runs are
# unchanged. Enable per-run via env (PREORDER_CALIBRATE=1) or constructor
# (Estimator(name, calibrate=True)). The A/B/C/D ablation in
# scripts/ablation_base_learner.py decides whether to flip the default.
CALIBRATE_PROBAS = os.environ.get("PREORDER_CALIBRATE", "0") not in ("0", "false", "False")
CALIBRATION_METHOD = os.environ.get("PREORDER_CALIBRATION_METHOD", "isotonic")
CALIBRATION_CV = int(os.environ.get("PREORDER_CALIBRATION_CV", "3"))


class Estimator:
    def __init__(self, name: str, calibrate: bool | None = None):
        self.name = name
        self.calibrate = CALIBRATE_PROBAS if calibrate is None else calibrate
        base = self.get_classifier()
        if self.calibrate:
            self.clf = CalibratedClassifierCV(
                base, method=CALIBRATION_METHOD, cv=CALIBRATION_CV
            )
        else:
            self.clf = base

    def get_classifier(self) -> BaseEstimator | LGBMClassifier:
        """Get the classifier based on name with proper error handling."""
        if self.name == BaseLearnerName.RF.value:
            return RandomForestClassifier(random_state=RANDOM_STATE)
        elif self.name == BaseLearnerName.ETC.value:
            return ExtraTreesClassifier(random_state=RANDOM_STATE)
        elif self.name == BaseLearnerName.XGBoost.value:
            return GradientBoostingClassifier(random_state=RANDOM_STATE)
        elif self.name == BaseLearnerName.LightGBM.value:
            return LGBMClassifier(
                random_state=RANDOM_STATE,
                n_jobs=int(number_of_cores - 1),
                # n_jobs=16,
                verbose=-1,
                num_leaves=20,  # Moderate complexity
                max_depth=6,
                bagging_fraction=0.9,
                feature_fraction=0.8,
                learning_rate=0.1,
                n_estimators=100,
                min_child_samples=5,  # Relaxed for small datasets
                min_child_weight=0.0001,  # Allow splits with low Hessian
                min_split_gain=0.01,  # Allow minimal gain splits
                is_unbalance=True,  # Handle label imbalance
                device="cpu",
            )
        else:
            raise ValueError(f"Unknown base learner: {self.name}")

    def fit(self, X: NDArray, Y: NDArray):
        """Fit the classifier with proper error handling.

        When calibration is enabled but the data is degenerate (single class
        or too few samples per class to support cv folds), fall back to the
        uncalibrated base estimator so the pairwise pipeline still runs.
        """
        try:
            self.clf.fit(X, Y)  # type: ignore
        except Exception as e:
            if self.calibrate:
                fallback = self.get_classifier()
                try:
                    fallback.fit(X, Y)  # type: ignore
                    self.clf = fallback
                    return
                except Exception as e2:
                    raise ValueError(
                        f"Error training {self.name} (calibrated and uncalibrated both failed): {e2}"
                    ) from e2
            raise ValueError(f"Error training {self.name}: {e}") from e

    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict the probability of each class for each instance."""
        prob = self.clf.predict_proba(X)  # type: ignore

        return prob  # type: ignore

    def classes_(self) -> list[int]:
        return self.clf.classes_  # type: ignore

    def predict(self, X: NDArray) -> NDArray:
        return self.clf.predict(X)  # type: ignore


def train_classifier(X, Y, estimator_name):
    classifier = Estimator(estimator_name)  # Add n_jobs or other params here
    classifier.fit(X, Y)
    return classifier
