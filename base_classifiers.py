from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb


class base_classifiers:
    def __init__(self, base_learner: str):
        self.base_learner = base_learner

    def save_model(model, filename):
        import pickle

        with open(filename, "wb") as f:
            pickle.dump(model, f)

    def _2classifier(self, X, Y):
        if self == "RF":
            clf = RandomForestClassifier(random_state=42)
        if self == "ET":
            clf = ExtraTreesClassifier(random_state=42)
        if self == "XGBoost":
            clf = GradientBoostingClassifier(random_state=42)
        if self == "LightGBM":
            clf = lgb.LGBMClassifier(random_state=42)
        #           features = lgb.Dataset(features, label= classes, categorical_feature=[0,1,4,5])
        #           print("check")
        #           print(features)
        clf.fit(X, Y)
        return clf
        # calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv="prefit")
        # calibrated_clf.fit(X, Y)
        # # sm = SMOTE(random_state=42)
        # # X_res, y_res = sm.fit_resample(features, classes)
        # # clf.fit(X_res, y_res)
        # return calibrated_clf

    def _3classifier(self, X, Y):
        if self == "RF":
            clf = RandomForestClassifier(random_state=42)
        if self == "ET":
            clf = ExtraTreesClassifier(random_state=42)
        if self == "XGBoost":
            clf = GradientBoostingClassifier(random_state=42)
        if self == "LightGBM":
            clf = lgb.LGBMClassifier(random_state=42)
        clf.fit(X, Y)
        return clf
        # calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv="prefit")
        # calibrated_clf.fit(X, Y)
        # # sm = SMOTE(random_state=42)
        # # X_res, y_res = sm.fit_resample(features, classes)
        # # clf.fit(X_res, y_res)
        # return calibrated_clf

    def _4classifier(self, X, Y):
        if self == "RF":
            clf = RandomForestClassifier(random_state=42)
        if self == "ET":
            clf = ExtraTreesClassifier(random_state=42)
        if self == "XGBoost":
            clf = GradientBoostingClassifier(random_state=42)
        if self == "LightGBM":
            clf = lgb.LGBMClassifier(random_state=42)
        clf.fit(X, Y)
        return clf
        # calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv="prefit")
        # calibrated_clf.fit(X, Y)
        # # sm = SMOTE(random_state=42)
        # # X_res, y_res = sm.fit_resample(features, classes)
        # # clf.fit(X_res, y_res)
        # return calibrated_clf
