from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from sklearn.base import BaseEstimator


class Estimator:
    def __init__(self, name: str):
        self.name = name

        if self.name == "RF":
            clf = RandomForestClassifier(random_state=42)
        elif self.name == "ET":
            clf = ExtraTreesClassifier(random_state=42)
        elif self.name == "XGBoost":
            clf = GradientBoostingClassifier(random_state=42)
        elif self.name == "LightGBM":
            clf = lgb.LGBMClassifier(random_state=42)
        self.clf: BaseEstimator = clf  # type: ignore

    def fit(self, X, Y) -> BaseEstimator:
        self.clf.fit(X, Y)  # type: ignore
        return self.clf

    # def _2classifier(self, X, Y):
    #     if self.name == "RF":
    #         clf = RandomForestClassifier(random_state=42)
    #     if self.name == "ET":
    #         clf = ExtraTreesClassifier(random_state=42)
    #     if self.name == "XGBoost":
    #         clf = GradientBoostingClassifier(random_state=42)
    #     if self.name == "LightGBM":
    #         clf = lgb.LGBMClassifier(random_state=42)
    #     clf.fit(X, Y)
    #     return clf

    # def _3classifier(self, X, Y):
    #     if self.name == "RF":
    #         clf = RandomForestClassifier(random_state=42)
    #     if self.name == "ET":
    #         clf = ExtraTreesClassifier(random_state=42)
    #     if self.name == "XGBoost":
    #         clf = GradientBoostingClassifier(random_state=42)
    #     if self.name == "LightGBM":
    #         clf = lgb.LGBMClassifier(random_state=42)
    #     clf.fit(X, Y)
    #     return clf

    # def _4classifier(self, X, Y):
    #     if self.name == "RF":
    #         clf = RandomForestClassifier(random_state=42)
    #     if self.name == "ET":
    #         clf = ExtraTreesClassifier(random_state=42)
    #     if self.name == "XGBoost":
    #         clf = GradientBoostingClassifier(random_state=42)
    #     if self.name == "LightGBM":
    #         clf = lgb.LGBMClassifier(random_state=42)
    #     clf.fit(X, Y)
    #     return clf
