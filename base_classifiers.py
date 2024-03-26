from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb


class BaseClassifiers:
    def __init__(self, name: str):
        self.name = name

    def save_model(self, model, filename):
        import pickle

        with open(filename, "wb") as f:
            pickle.dump(model, f)

    def _2classifier(self, X, Y):
        if self.name == "RF":
            clf = RandomForestClassifier(random_state=42)
        if self.name == "ET":
            clf = ExtraTreesClassifier(random_state=42)
        if self.name == "XGBoost":
            clf = GradientBoostingClassifier(random_state=42)
        if self.name == "LightGBM":
            clf = lgb.LGBMClassifier(random_state=42)
        clf.fit(X, Y)
        return clf

    def _3classifier(self, X, Y):
        if self.name == "RF":
            clf = RandomForestClassifier(random_state=42)
        if self.name == "ET":
            clf = ExtraTreesClassifier(random_state=42)
        if self.name == "XGBoost":
            clf = GradientBoostingClassifier(random_state=42)
        if self.name == "LightGBM":
            clf = lgb.LGBMClassifier(random_state=42)
        clf.fit(X, Y)
        return clf

    def _4classifier(self, X, Y):
        if self.name == "RF":
            clf = RandomForestClassifier(random_state=42)
        if self.name == "ET":
            clf = ExtraTreesClassifier(random_state=42)
        if self.name == "XGBoost":
            clf = GradientBoostingClassifier(random_state=42)
        if self.name == "LightGBM":
            clf = lgb.LGBMClassifier(random_state=42)
        clf.fit(X, Y)
        return clf
