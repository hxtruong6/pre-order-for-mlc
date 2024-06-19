import numpy as np
from enum import Enum
from base_classifiers import BaseClassifiers
from constants import TargetMetric
from estimator import Estimator
from searching_algorithms import Search_BOPreOs, Search_BOParOs


class PreferenceOrder(Enum):
    PRE_ORDER = "PreOrder"
    BIPARTITE_PRE_ORDER = "BipartitePreOrder"
    PARTIAL_ORDER = "PartialOrder"
    BIPARTITE_PARTIAL_ORDER = "BipartitePartialOrder"

    PRE_ORDER_HAM = "PreOrderHam"
    BIPARTITE_PRE_ORDER_HAM = "BipartitePreOrderHam"
    PARTIAL_ORDER_HAM = "PartialOrderHam"
    BIPARTITE_PARTIAL_ORDER_HAM = "BipartitePartialOrderHam"

    PRE_ORDER_SUB = "PreOrderSub"
    BIPARTITE_PRE_ORDER_SUB = "BipartitePreOrderSub"
    PARTIAL_ORDER_SUB = "PartialOrderSub"
    BIPARTITE_PARTIAL_ORDER_SUB = "BipartitePartialOrderSub"


class PredictBOPOs:
    def __init__(
        self,
        estimator: Estimator,
        preference_order: PreferenceOrder = PreferenceOrder.PRE_ORDER,
    ):
        # BaseClassifiers is a class that contains the base learner (estimator) to train the model with input X and predicted labels Y
        self.base_classifier = BaseClassifiers(estimator)

        self.preference_order = preference_order

        self.models = {}

        self.pairwise_classifier = None

    def predict_preference_orders(
        self,
        pairwise_probabilistic_predictions,
        n_labels,
        n_instances,
        target_metric: TargetMetric,
    ):
        # TODO: Refactor this func later

        # 1. Initialize a search BOPreOs model
        search_BOPrerOs = Search_BOPreOs(
            pairwise_probabilistic_predictions,
            n_labels,
            n_instances,
            target_metric,
            height=2,
        )

        # 2. Fit the model with the input data
        # Placeholder for prediction process
        if self.preference_order == PreferenceOrder.PRE_ORDER_HAM:
            predict_BOPOS, predict_binary_vectors = search_BOPrerOs.PRE_ORDER(
                pairwise_probabilistic_predictions, n_labels, n_instances, target_metric
            )
        elif self.preference_order == PreferenceOrder.BIPARTITE_PRE_ORDER_HAM:
            predict_BOPOS, predict_binary_vectors = search_BOPrerOs.PRE_ORDER(
                pairwise_probabilistic_predictions,
                n_labels,
                n_instances,
                target_metric,
                height=2,
            )
        elif self.preference_order == PreferenceOrder.PRE_ORDER_SUB:
            predict_BOPOS, predict_binary_vectors = search_BOPrerOs.PRE_ORDER(
                pairwise_probabilistic_predictions, n_labels, n_instances, target_metric
            )
        elif self.preference_order == PreferenceOrder.BIPARTITE_PRE_ORDER_SUB:
            predict_BOPOS, predict_binary_vectors = search_BOPrerOs.PRE_ORDER(
                pairwise_probabilistic_predictions,
                n_labels,
                n_instances,
                target_metric,
                height=2,
            )
        else:
            search_BOParOs = Search_BOParOs(
                pairwise_probabilistic_predictions,
                n_labels,
                n_instances,
                target_metric,
                height=2,
            )

            if self.preference_order == PreferenceOrder.PARTIAL_ORDER_HAM:
                predict_BOPOS, predict_binary_vectors = search_BOParOs.PARTIAL_ORDER(
                    pairwise_probabilistic_predictions,
                    n_labels,
                    n_instances,
                    target_metric,
                )
            elif self.preference_order == PreferenceOrder.BIPARTITE_PARTIAL_ORDER_HAM:
                predict_BOPOS, predict_binary_vectors = search_BOParOs.PARTIAL_ORDER(
                    pairwise_probabilistic_predictions,
                    n_labels,
                    n_instances,
                    target_metric,
                    height=2,
                )
            elif self.preference_order == PreferenceOrder.PARTIAL_ORDER_SUB:
                predict_BOPOS, predict_binary_vectors = search_BOParOs.PARTIAL_ORDER(
                    pairwise_probabilistic_predictions,
                    n_labels,
                    n_instances,
                    target_metric,
                )
            elif self.preference_order == PreferenceOrder.BIPARTITE_PARTIAL_ORDER_SUB:
                predict_BOPOS, predict_binary_vectors = search_BOParOs.PARTIAL_ORDER(
                    pairwise_probabilistic_predictions,
                    n_labels,
                    n_instances,
                    target_metric,
                    height=2,
                )
            else:
                raise ValueError(f"Unknown preference order: {self.preference_order}")

        # 3. Return the predicted preference orders and predicted labels

        # it should return the predicted preference orders and predicted labels predicted_Y
        # return predicted_O, predicted_Y
        return predict_BOPOS, predict_binary_vectors

    def predict_proba(self, X, n_labels):
        n_test_instances, _ = X.shape
        # Placeholder for prediction process
        if (
            self.preference_order == PreferenceOrder.PRE_ORDER
            or self.preference_order == PreferenceOrder.BIPARTITE_PRE_ORDER
        ):
            pairwise_probabilistic_predictions = {}
            for i in range(n_labels - 1):
                for j in range(i + 1, n_labels):
                    pairwise_probabilistic_predictions = np.zeros((n_test_instances, 4))
                    key_classifier = f"{i}_{j}"

                    # TODO: recheck this line
                    original_pairwise_probabilistic_predictions = (
                        self.pairwise_classifier[key_classifier](X)
                    )

                    # TODO: "classes_" is not defined. Check type of pairwise_classifier later
                    presented_classes = list(self.pairwise_classifier.classes_)
                    for l in range(4):
                        if l in presented_classes:
                            pairwise_probabilistic_predictions[:, l] = (
                                original_pairwise_probabilistic_predictions[
                                    :, presented_classes.index(l)
                                ]
                            )
                    for n in range(n_test_instances):

                        # add a small regularization term if the probabilistic prediction is deterministic instead of probabilistic
                        current_pairwise_probabilistic_predictions = (
                            pairwise_probabilistic_predictions[n]
                        )
                        if max(current_pairwise_probabilistic_predictions) == 1:
                            current_pairwise_probabilistic_predictions = [
                                x - 10**-10 if x == 1 else (10**-10) / 3
                                for x in current_pairwise_probabilistic_predictions
                            ]
                        if min(current_pairwise_probabilistic_predictions) == 0:
                            zero_indices = [
                                ind
                                for ind in range(4)
                                if current_pairwise_probabilistic_predictions[ind] == 0
                            ]
                            current_pairwise_probabilistic_predictions = [
                                (
                                    (10**-10) / len(zero_indices)
                                    if x == 0
                                    else x - (10**-10) / (4 - len(zero_indices))
                                )
                                for x in current_pairwise_probabilistic_predictions
                            ]
                        for l in range(4):
                            #                            key_pairwise_probabilistic_predictions = "%i_%i_%i_%i" % (i, j, n,l)
                            pairwise_probabilistic_predictions[f"{i}_{j}_{n}_{l}"] = (
                                current_pairwise_probabilistic_predictions[l]
                            )
        elif (
            self.preference_order == PreferenceOrder.PARTIAL_ORDER
            or self.preference_order == PreferenceOrder.BIPARTITE_PARTIAL_ORDER
        ):
            pairwise_probabilistic_predictions = {}
            for i in range(n_labels - 1):
                for j in range(i + 1, n_labels):
                    pairwise_probabilistic_predictions = np.zeros((n_test_instances, 3))
                    key_classifier = f"{i}_{j}"
                    original_pairwise_probabilistic_predictions = (
                        self.pairwise_classifier[key_classifier](X)
                    )
                    presented_classes = list(self.pairwise_classifier.classes_)
                    for l in range(3):
                        if l in presented_classes:
                            pairwise_probabilistic_predictions[:, l] = (
                                original_pairwise_probabilistic_predictions[
                                    :, presented_classes.index(l)
                                ]
                            )
                    for n in range(n_test_instances):

                        # add a small regularization term if the probabilistic prediction is deterministic instead of probabilistic
                        current_pairwise_probabilistic_predictions = (
                            pairwise_probabilistic_predictions[n]
                        )
                        if max(current_pairwise_probabilistic_predictions) == 1:
                            current_pairwise_probabilistic_predictions = [
                                x - 10**-10 if x == 1 else (10**-10) / 2
                                for x in current_pairwise_probabilistic_predictions
                            ]
                        if min(current_pairwise_probabilistic_predictions) == 0:
                            zero_indices = [
                                ind
                                for ind in range(3)
                                if current_pairwise_probabilistic_predictions[ind] == 0
                            ]
                            current_pairwise_probabilistic_predictions = [
                                (
                                    (10**-10) / len(zero_indices)
                                    if x == 0
                                    else x - (10**-10) / (3 - len(zero_indices))
                                )
                                for x in current_pairwise_probabilistic_predictions
                            ]
                        for l in range(3):
                            #                            key_pairwise_probabilsitic_predictions = "%i_%i_%i_%i" % (i, j, n,l)
                            pairwise_probabilistic_predictions[f"{i}_{j}_{n}_{l}"] = (
                                current_pairwise_probabilistic_predictions[l]
                            )
        return pairwise_probabilistic_predictions

    #

    def fit(self, X, Y):
        """Training the model

        Args:
            X (_type_): _description_
            Y (_type_): _description_

        Raises:
            ValueError: _description_
        """
        if self.preference_order == PreferenceOrder.PRE_ORDER:
            self.pairwise_classifier = (
                self.base_classifier.pairwise_pre_order_classifier(X, Y)
            )
            pass
        elif self.preference_order == PreferenceOrder.PARTIAL_ORDER:
            self.pairwise_classifier = (
                self.base_classifier.pairwise_partial_order_classifer(X, Y)
            )
            pass
        else:
            raise ValueError(f"Unknown preference order: {self.preference_order}")


#  These functions can be moved to another separate class
#    def classifier_chains(self, *args, **kwargs):
#        chains = [
#            ClassifierChain(self.estimator, order="random", random_state=i)  # type: ignore
#            for i in range(10)  # TODO: what is 10?
#        ]
#        # for chain in chains:
#        #     chain.fit(X[train_index], Y[train_index])
#
#        # Y_pred_chains = np.array([chain.predict(X[test_index]) for chain in chains])
#        # Y_pred_ensemble = Y_pred_chains.mean(axis=0)
#        # predicted_Y = np.where(Y_pred_ensemble > 0.5, 1, 0)
#
#    def calibrated_label_ranking(self, *args, **kwargs):
#        chains = [
#            ClassifierChain(self.estimator, order="random", random_state=i)  # type: ignore
#            for i in range(10)  # TODO: what is 10?
#        ]
#        # for chain in chains:
#        #     chain.fit(X[train_index], Y[train_index])
#
#        # Y_pred_chains = np.array([chain.predict(X[test_index]) for chain in chains])
#        # Y_pred_ensemble = Y_pred_chains.mean(axis=0)
#        # predicted_Y = np.where(Y_pred_ensemble > 0.5, 1, 0)
