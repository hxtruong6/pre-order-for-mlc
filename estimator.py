from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from sklearn.base import BaseEstimator
from numpy.typing import NDArray
from lightgbm import LGBMClassifier

from constants import RANDOM_STATE


class Estimator:
    def __init__(self, name: str):
        self.name = name
        self.clf: BaseEstimator | LGBMClassifier = self.get_classifier()

    def get_classifier(self) -> BaseEstimator | LGBMClassifier:
        """Get the classifier based on name with proper error handling."""
        try:
            if self.name == "RF":
                return RandomForestClassifier(random_state=RANDOM_STATE)
            elif self.name == "ET":
                return ExtraTreesClassifier(random_state=RANDOM_STATE)
            elif self.name == "XGBoost":
                return GradientBoostingClassifier(random_state=RANDOM_STATE)
            elif self.name == "LightGBM":
                # TODO: check this
                return LGBMClassifier(random_state=RANDOM_STATE)
            else:
                raise ValueError(f"Unknown base learner: {self.name}")
        except Exception as e:
            raise ValueError(f"Error initializing {self.name}: {str(e)}")

    def fit(self, X: NDArray, Y: NDArray) -> BaseEstimator | LGBMClassifier:
        """Fit the classifier with proper error handling."""
        try:
            if self.clf is None:
                self.clf = self.get_classifier()
            assert isinstance(self.clf, BaseEstimator | LGBMClassifier)

            return self.clf.fit(X, Y)
        except Exception as e:
            raise ValueError(f"Error training {self.name}: {str(e)}")
