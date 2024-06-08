from enum import Enum
from sklearn.base import BaseEstimator
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from base_classifer import BaseClassifiers
import lightgbm
import numpy as np


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

    def prob_predict(self, X):
        n_test_instances, _ = X.shape
        # Placeholder for prediction process
        if self.preference_order == PreferenceOrder.PRE_ORDER or self.preference_order == PreferenceOrder.BIPARTITE_PRE_ORDER:
            pairwise_probabilistic_predictions = {}
            for i in range(n_labels - 1):
                for j in range(i + 1, n_labels):
                    pairwise_probabilistic_predictions = np.zeros((n_test_instances,3))
                    key_classifier = f"{i}_{j}"
                    original_pairwise_probabilistic_predictions = self.pairwise_classifier[key_classifier](X)
                    presented_classes = list(self.pairwise_classifier.classes_)
                    for l in range(4):
                        if l in presented_classes:
                            pairwise_probabilistic_predictions[:,l] = original_pairwise_probabilistic_predictions[:,presented_classes.index(l)]                             
                    for n in range(n_test_instances):
                                        
                        # add a small regularization term if the probabilistic prediction is deterministic instead of probabilistic

                        for l in range(4)
                            key_pairwise_probabilsitic_predictions = "%i_%i_%i_%i" % (i, j, n,l)
                            pairwise_probabilsitic_predictions[key_pairwise_probabilsitic_predictions] = pairwise_probabilistic_predictions[n]                 
            pass
        elif self.preference_order == PreferenceOrder.PARTIAL_ORDER or self.preference_order == PreferenceOrder.BIPARTITE_PARTIAL_ORDER:
            pairwise_probabilistic_predictions = {}
            for i in range(n_labels - 1):
                for j in range(i + 1, n_labels):
                    pairwise_probabilistic_predictions = np.zeros((n_test_instances,3))
                    key_classifier = f"{i}_{j}"
                    original_pairwise_probabilistic_predictions = self.pairwise_classifier[key_classifier](X)
                    presented_classes = list(self.pairwise_classifier.classes_)
                    for l in range(3):
                        if l in presented_classes:
                            pairwise_probabilistic_predictions[:,l] = original_pairwise_probabilistic_predictions[:,presented_classes.index(l)]                             
                    for n in range(n_test_instances):

                        # add a small regularization term if the probabilistic prediction is deterministic instead of probabilistic

                        for l in range(3)
                            key_pairwise_probabilsitic_predictions = "%i_%i_%i_%i" % (i, j, n,l)
                            pairwise_probabilsitic_predictions[key_pairwise_probabilsitic_predictions] = pairwise_probabilistic_predictions[n]
            pass
        return pairwise_probabilsitic_predictions

    #   

    def fit(self, X, Y):
        # TODO
        if self.preference_order == PreferenceOrder.PRE_ORDER:
            self.pairwise_classifier = BaseClassifiers.pairwise_pre_order_classifier(X, Y)
            pass
        elif self.preference_order == PreferenceOrder.PARTIAL_ORDER:
            self.pairwise_classifier = BaseClassifiers.pairwise_partial_order_classifer(X, Y)
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
