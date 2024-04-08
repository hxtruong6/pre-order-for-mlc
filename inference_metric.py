from enum import Enum
from sklearn.base import BaseEstimator
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm


class PreferenceOrder(Enum):
    PRE_ORDER = "PreOrder"
    PARTIAL_ORDER = "PartialOrder"


class InferenceMetric:
    def __init__(
        self,
        estimator: BaseEstimator | lightgbm.LGBMClassifier,
    ):
        self.estimator = estimator
        self.preference_orders = [
            PreferenceOrder.PRE_ORDER,
            PreferenceOrder.PARTIAL_ORDER,
        ]
        self.result = {
            f"{PreferenceOrder.PRE_ORDER}": {},
            f"{PreferenceOrder.PARTIAL_ORDER}": {},
        }

        self.models = {}

    def predict(self, X):
        # Placeholder for prediction process
        pass

    def process_training(self, X, Y):
        for preference_order in self.preference_orders:
            # Apply preference order: PreOrder or PartialOrder
            y_hamming = self._hamming(X, Y)
            y_weighted_hamming = self._weighted_hamming(X, Y)
            y_subset = self._subset(X, Y)

            self.result[f"{preference_order}"]["hamming"] = y_hamming
            self.result[f"{preference_order}"]["weighted_hamming"] = y_weighted_hamming
            self.result[f"{preference_order}"]["subset"] = y_subset

    # Placeholder for metric calculation methods
    def _hamming(self, *args, **kwargs):
        for preference_order in self.preference_orders:
            # Apply preference order: PreOrder or PartialOrder
            pass

    def _weighted_hamming(self, *args, **kwargs):
        pass

    def _subset(self, *args, **kwargs):
        pass

    def _classify_chain(self, *args, **kwargs):
        chains = [
            ClassifierChain(self.estimator, order="random", random_state=i)  # type: ignore
            for i in range(10)  # TODO: what is 10?
        ]
        # for chain in chains:
        #     chain.fit(X[train_index], Y[train_index])

        # Y_pred_chains = np.array([chain.predict(X[test_index]) for chain in chains])
        # Y_pred_ensemble = Y_pred_chains.mean(axis=0)
        # predicted_Y = np.where(Y_pred_ensemble > 0.5, 1, 0)
