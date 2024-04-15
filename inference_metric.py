from enum import Enum
from sklearn.base import BaseEstimator
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm


class PreferenceOrder(Enum):
    PRE_ORDER = "PreOrder"
    BIPARTITE_PRE_ORDER = "BipartitePreOrder"
    PARTIAL_ORDER = "PartialOrder"
    BIPARTITE_PARTIAL_ORDER = "BipartitePartialOrder"


class InferenceMetric:
    def __init__(
        self,
        estimator: BaseEstimator | lightgbm.LGBMClassifier,
        preference_order: PreferenceOrder = PreferenceOrder.PRE_ORDER,
    ):
        self.estimator = estimator
        self.preference_order = preference_order

        self.models = {}

        self.pairwise_classifier = None

    def predict(self, X):
        # Placeholder for prediction process
        if self.preference_order == PreferenceOrder.PRE_ORDER:
            pass
        elif self.preference_order == PreferenceOrder.BIPARTITE_PRE_ORDER:
            pass
        elif self.preference_order == PreferenceOrder.PARTIAL_ORDER:
            pass
        elif self.preference_order == PreferenceOrder.BIPARTITE_PARTIAL_ORDER:
            pass
        # it should return the predicted preference orders and predicted labels predicted_Y
        # return predicted_O, predicted_Y

    def fit(self, X, Y):
        # TODO
        if self.preference_order == PreferenceOrder.PRE_ORDER:
            # self.pairwise_classifier = self._pairwise_2classifier(X, Y)
            pass
        elif self.preference_order == PreferenceOrder.PARTIAL_ORDER:
            # self.pairwise_classifier = self._pairwise_3classifier(X, Y)
            pass
        else:
            raise ValueError(f"Unknown preference order: {self.preference_order}")

    # Placeholder for metric calculation methods
    def _hamming_PL(self, *args, **kwargs):
        # Apply preference order: PreOrder or PartialOrder
        pass

    def _weighted_hamming_PL(self, *args, **kwargs):
        pass

    def _subset_PL(self, *args, **kwargs):
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
