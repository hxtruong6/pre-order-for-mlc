from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from sklearn.base import BaseEstimator

from constants import RANDOM_STATE


class Estimator:
    def __init__(self, name: str):
        self.name = name

        if self.name == "RF":
            clf = RandomForestClassifier(random_state=RANDOM_STATE)
        elif self.name == "ET":
            clf = ExtraTreesClassifier(random_state=RANDOM_STATE)
        elif self.name == "XGBoost":
            clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
        elif self.name == "LightGBM":
            clf = lgb.LGBMClassifier(random_state=RANDOM_STATE)
        else:
            raise ValueError(f"Unknown base learner: {self.name}")
        self.clf: BaseEstimator | lgb.LGBMClassifier = clf  # type: ignore

    def fit(self, X, Y):
        self.clf.fit(X, Y)  # type: ignore
        return self.clf

    # def _2classifier(self, X, Y):
    #     if self.name == "RF":
    #         clf = RandomForestClassifier(random_state=RANDOM_STATE)
    #     if self.name == "ET":
    #         clf = ExtraTreesClassifier(random_state=RANDOM_STATE)
    #     if self.name == "XGBoost":
    #         clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    #     if self.name == "LightGBM":
    #         clf = lgb.LGBMClassifier(random_state=RANDOM_STATE)
    #     clf.fit(X, Y)
    #     return clf

    # def _3classifier(self, X, Y):
    #     if self.name == "RF":
    #         clf = RandomForestClassifier(random_state=RANDOM_STATE)
    #     if self.name == "ET":
    #         clf = ExtraTreesClassifier(random_state=RANDOM_STATE)
    #     if self.name == "XGBoost":
    #         clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    #     if self.name == "LightGBM":
    #         clf = lgb.LGBMClassifier(random_state=RANDOM_STATE)
    #     clf.fit(X, Y)
    #     return clf

    # def _4classifier(self, X, Y):
    #     if self.name == "RF":
    #         clf = RandomForestClassifier(random_state=RANDOM_STATE)
    #     if self.name == "ET":
    #         clf = ExtraTreesClassifier(random_state=RANDOM_STATE)
    #     if self.name == "XGBoost":
    #         clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    #     if self.name == "LightGBM":
    #         clf = lgb.LGBMClassifier(random_state=RANDOM_STATE)
    #     clf.fit(X, Y)
    #     return clf
