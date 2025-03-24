from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator
from numpy.typing import NDArray
from lightgbm import LGBMClassifier

from constants import RANDOM_STATE, BaseLearnerName


class Estimator:
    def __init__(self, name: str):
        self.name = name
        self.clf: BaseEstimator | LGBMClassifier = self.get_classifier()

    def get_classifier(self) -> BaseEstimator | LGBMClassifier:
        """Get the classifier based on name with proper error handling."""
        if self.name == BaseLearnerName.RF.value:
            return RandomForestClassifier(random_state=RANDOM_STATE)
        elif self.name == BaseLearnerName.ETC.value:
            return ExtraTreesClassifier(random_state=RANDOM_STATE)
        elif self.name == BaseLearnerName.XGBoost.value:
            return GradientBoostingClassifier(random_state=RANDOM_STATE)
        elif self.name == BaseLearnerName.LightGBM.value:
            return LGBMClassifier(random_state=RANDOM_STATE)
        else:
            raise ValueError(f"Unknown base learner: {self.name}")

    def fit(self, X: NDArray, Y: NDArray):
        """Fit the classifier with proper error handling."""
        try:
            assert isinstance(self.clf, BaseEstimator | LGBMClassifier)
            self.clf.fit(X, Y)  # type: ignore
        except Exception as e:
            raise ValueError(f"Error training {self.name}: {str(e)}")

    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict the probability of each class for each instance."""
        prob = self.clf.predict_proba(X)  # type: ignore

        return prob  # type: ignore

    def classes_(self) -> list[int]:
        return self.clf.classes_  # type: ignore
