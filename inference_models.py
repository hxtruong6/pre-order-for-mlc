from logging import INFO, log
from joblib import Parallel, delayed
import numpy as np
from enum import Enum
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain

from base_classifiers import BaseClassifiers
from constants import BaseLearnerName, TargetMetric, RANDOM_STATE
from estimator import Estimator
from searching_algorithms import Search_BOPreOs, Search_BOParOs


class PreferenceOrder(Enum):
    PRE_ORDER = "PreOrder"
    PARTIAL_ORDER = "PartialOrder"


"""
1. Predict probability
2. Predict preference orders
"""


class PredictBOPOs:

    def __init__(
        self,
        base_classifier_name: str,
        preference_order: PreferenceOrder = PreferenceOrder.PRE_ORDER,
    ):
        # BaseClassifiers is a class that contains the base learner (estimator)
        # to train the model with input X and predicted labels Y
        self.base_classifier: BaseClassifiers = BaseClassifiers(base_classifier_name)

        self.preference_order = preference_order

        self.models = {}

        self.pairwise_classifier: dict[str, Estimator] = {}

        self.calibrated_classifier: list[Estimator] = []

        self.single_label_pair: dict[str, int | None] = {  # key = label i_j
            # 1_2: 1
            # 1_3: 0
            # 1_4: None
        }

        # Add new attributes for BR and CC
        self.br_classifiers: dict[str, Estimator] = {}  # Binary Relevance classifiers
        self.cc_classifiers: dict[str, Estimator] = {}  # Classifier Chain classifiers
        self.cc_order: list[int] = []  # Order of labels for Classifier Chain

    def predict_preference_orders(
        self,
        pairwise_probabilistic_predictions,
        n_labels,
        n_instances,
        target_metric: TargetMetric,
        height: int | None = None,
    ) -> tuple[list[int], list[float], list[int] | None]:
        # 4 cases for pre-order and 4 cases for partial-order
        log(
            INFO,
            f"--Target metric: {target_metric}, Preference order: {self.preference_order}, Height: {height}",
        )
        # 1. Initialize a search BOPreOs model
        search_BOPrerOs = Search_BOPreOs(
            pairwise_probabilistic_predictions,
            n_labels,
            n_instances,
            target_metric,
            height=height,
        )
        search_BOParOs = Search_BOParOs(
            pairwise_probabilistic_predictions,
            n_labels,
            n_instances,
            target_metric,
            height=height,
        )

        # Using after training the model.
        if target_metric == TargetMetric.Hamming:
            if self.preference_order == PreferenceOrder.PRE_ORDER:
                predict_BOPOS, predict_binary_vectors, indices_vector = (
                    search_BOPrerOs.PRE_ORDER()
                )
            elif self.preference_order == PreferenceOrder.PARTIAL_ORDER:
                predict_BOPOS, predict_binary_vectors, indices_vector = (
                    search_BOParOs.PARTIAL_ORDER()
                )
            else:
                raise ValueError(
                    f"[Hamming] Unknown preference order: {self.preference_order}"
                )

        elif target_metric == TargetMetric.Subset:
            if self.preference_order == PreferenceOrder.PRE_ORDER:
                predict_BOPOS, predict_binary_vectors, indices_vector = (
                    search_BOPrerOs.PRE_ORDER()
                )
            elif self.preference_order == PreferenceOrder.PARTIAL_ORDER:
                predict_BOPOS, predict_binary_vectors, indices_vector = (
                    search_BOParOs.PARTIAL_ORDER()
                )

            else:
                raise ValueError(
                    f"[Subset] Unknown preference order: {self.preference_order}"
                )

        return predict_BOPOS, predict_binary_vectors, indices_vector  # type: ignore

    def predict_proba(self, X, n_labels):
        n_test_instances, _ = X.shape
        # Placeholder for prediction process
        if (
            self.preference_order
            == PreferenceOrder.PRE_ORDER
            # or self.preference_order == PreferenceOrder.BIPARTITE_PRE_ORDER
        ):

            def _get_pairwise_predict_proba():
                predict_results = Parallel(n_jobs=-1)(
                    delayed(self.pairwise_classifier[f"{i}_{j}"].predict_proba)(X)
                    for i in range(n_labels - 1)
                    for j in range(i + 1, n_labels)
                )

                return dict(zip(self.pairwise_classifier.keys(), predict_results))

            pairwise_probabilistic_predictions = {}
            pairwise_predict_proba = _get_pairwise_predict_proba()
            for i in range(n_labels - 1):
                for j in range(i + 1, n_labels):
                    pairwise_probabilistic_predictions_ij = np.zeros(
                        (n_test_instances, 4)
                    )
                    key_classifier = f"{i}_{j}"

                    # TODO: recheck this line. DEBUG this predict_proba function.
                    # Output: probability predictions. Value is a matrix [test_instances, 4] PRE_ORDER. PARTIAL_ORDER. [test_instance, 3]
                    # original_pairwise_probabilistic_predictions_ij = (
                    #     self.pairwise_classifier[key_classifier].predict_proba(X)
                    # )
                    original_pairwise_probabilistic_predictions_ij = (
                        pairwise_predict_proba[key_classifier]
                    )
                    # original_pairwise_probabilistic_predictions_ij = [test_instances, 1/2/3/4]. Not guaranteed 4/3 classes.

                    # TODO: "classes_" is not defined. Check type of pairwise_classifier later
                    presented_classes = list(
                        self.pairwise_classifier[key_classifier].classes_()
                    )
                    # Output: maxtrix [test_instances, 4] PRE_ORDER. PARTIAL_ORDER. [test_instance, 3]
                    for l in range(4):
                        if l in presented_classes:
                            pairwise_probabilistic_predictions_ij[:, l] = (
                                # original_pairwise_probabilistic_predictions_ij[
                                #     :, presented_classes.index(l)
                                # ]
                                pairwise_predict_proba[key_classifier][:, l]  # type: ignore
                            )
                    for n in range(n_test_instances):

                        # add a small regularization term if the probabilistic prediction is deterministic instead of probabilistic
                        current_pairwise_probabilistic_predictions_ij = (
                            pairwise_probabilistic_predictions_ij[n]
                        )
                        if max(current_pairwise_probabilistic_predictions_ij) == 1:
                            current_pairwise_probabilistic_predictions_ij = [
                                x - 10**-10 if x == 1 else (10**-10) / 3
                                for x in current_pairwise_probabilistic_predictions_ij
                            ]
                        if min(current_pairwise_probabilistic_predictions_ij) == 0:
                            zero_indices = [
                                ind
                                for ind in range(4)
                                if current_pairwise_probabilistic_predictions_ij[ind]
                                == 0
                            ]
                            current_pairwise_probabilistic_predictions_ij = [
                                (
                                    (10**-10) / len(zero_indices)
                                    if x == 0
                                    else x - (10**-10) / (4 - len(zero_indices))
                                )
                                for x in current_pairwise_probabilistic_predictions_ij
                            ]
                        for l in range(4):
                            pairwise_probabilistic_predictions[f"{i}_{j}_{n}_{l}"] = (
                                current_pairwise_probabilistic_predictions_ij[l]
                            )
        elif (
            self.preference_order
            == PreferenceOrder.PARTIAL_ORDER
            # or self.preference_order == PreferenceOrder.BIPARTITE_PARTIAL_ORDER
        ):

            def _get_pairwise_predict_proba():
                predict_results = Parallel(n_jobs=-1)(
                    delayed(self.pairwise_classifier[f"{i}_{j}"].predict_proba)(X)
                    for i in range(n_labels - 1)
                    for j in range(i + 1, n_labels)
                )

                return dict(zip(self.pairwise_classifier.keys(), predict_results))

            pairwise_probabilistic_predictions = {}
            pairwise_predict_proba = _get_pairwise_predict_proba()
            for i in range(n_labels - 1):
                for j in range(i + 1, n_labels):
                    pairwise_probabilistic_predictions_ij = np.zeros(
                        (n_test_instances, 3)
                    )
                    key_classifier = f"{i}_{j}"
                    # original_pairwise_probabilistic_predictions_ij = (
                    #     self.pairwise_classifier[key_classifier].predict_proba(X)
                    # )
                    original_pairwise_probabilistic_predictions_ij = (
                        pairwise_predict_proba[key_classifier]
                    )
                    presented_classes = list(
                        self.pairwise_classifier[key_classifier].classes_()
                    )
                    for l in range(3):
                        if l in presented_classes:
                            pairwise_probabilistic_predictions_ij[:, l] = (
                                # original_pairwise_probabilistic_predictions_ij[
                                #     :, presented_classes.index(l)
                                # ]
                                pairwise_predict_proba[key_classifier][:, l]  # type: ignore
                            )
                    for n in range(n_test_instances):

                        # add a small regularization term if the probabilistic prediction is deterministic instead of probabilistic
                        current_pairwise_probabilistic_predictions_ij = (
                            pairwise_probabilistic_predictions_ij[n]
                        )
                        if max(current_pairwise_probabilistic_predictions_ij) == 1:
                            current_pairwise_probabilistic_predictions_ij = [
                                x - 10**-10 if x == 1 else (10**-10) / 2
                                for x in current_pairwise_probabilistic_predictions_ij
                            ]
                        if min(current_pairwise_probabilistic_predictions_ij) == 0:
                            zero_indices = [
                                ind
                                for ind in range(3)
                                if current_pairwise_probabilistic_predictions_ij[ind]
                                == 0
                            ]
                            current_pairwise_probabilistic_predictions_ij = [
                                (
                                    (10**-10) / len(zero_indices)
                                    if x == 0
                                    else x - (10**-10) / (3 - len(zero_indices))
                                )
                                for x in current_pairwise_probabilistic_predictions_ij
                            ]
                        for l in range(3):
                            #                            key_pairwise_probabilistic_predictions = "%i_%i_%i_%i" % (i, j, n,l)
                            pairwise_probabilistic_predictions[f"{i}_{j}_{n}_{l}"] = (
                                current_pairwise_probabilistic_predictions_ij[l]
                            )
        return pairwise_probabilistic_predictions

    def predict_proba_BR(self, X, n_labels):
        # TODO We need to train a binary relevance classifier and store its K binary classifiers in a dictionary
        n_test_instances, _ = X.shape
        # Placeholder for prediction process
        if (
            self.preference_order
            == PreferenceOrder.PRE_ORDER
            # or self.preference_order == PreferenceOrder.BIPARTITE_PRE_ORDER
        ):
            pairwise_probabilistic_predictions = {}
            for i in range(n_labels - 1):
                key_classifier_i = f"{i}"
                original_probabilistic_predictions_i = self.pairwise_classifier[
                    key_classifier_i
                ].predict_proba(X)
                presented_classes = list(
                    self.pairwise_classifier[key_classifier_i].classes_()
                )
                probabilistic_predictions_i = np.zeros((n_test_instances, 2))
                for c in range(2):
                    if c in presented_classes:
                        probabilistic_predictions_i[:, c] = (
                            original_probabilistic_predictions_i[
                                :, presented_classes.index(c)
                            ]
                        )
                for j in range(i + 1, n_labels):
                    key_classifier_j = f"{j}"
                    original_probabilistic_predictions_j = self.pairwise_classifier[
                        key_classifier_j
                    ].predict_proba(X)
                    presented_classes = list(
                        self.pairwise_classifier[key_classifier_j].classes_()
                    )
                    probabilistic_predictions_j = np.zeros((n_test_instances, 2))
                    for c in range(2):
                        if c in presented_classes:
                            probabilistic_predictions_j[:, c] = (
                                original_probabilistic_predictions_j[
                                    :, presented_classes.index(c)
                                ]
                            )
                    for n in range(n_test_instances):

                        current_pairwise_probabilistic_predictions_ij = [
                            probabilistic_predictions_i[n, 1]
                            * probabilistic_predictions_j[n, 0],
                            probabilistic_predictions_i[n, 0]
                            * probabilistic_predictions_j[n, 1],
                            probabilistic_predictions_i[n, 0]
                            * probabilistic_predictions_j[n, 0],
                            probabilistic_predictions_i[n, 1]
                            * probabilistic_predictions_j[n, 1],
                        ]

                        # add a small regularization term if the probabilistic prediction is deterministic instead of probabilistic

                        if max(current_pairwise_probabilistic_predictions_ij) == 1:
                            current_pairwise_probabilistic_predictions_ij = [
                                x - 10**-10 if x == 1 else (10**-10) / 3
                                for x in current_pairwise_probabilistic_predictions_ij
                            ]
                        if min(current_pairwise_probabilistic_predictions_ij) == 0:
                            zero_indices = [
                                ind
                                for ind in range(4)
                                if current_pairwise_probabilistic_predictions_ij[ind]
                                == 0
                            ]
                            current_pairwise_probabilistic_predictions_ij = [
                                (
                                    (10**-10) / len(zero_indices)
                                    if x == 0
                                    else x - (10**-10) / (4 - len(zero_indices))
                                )
                                for x in current_pairwise_probabilistic_predictions_ij
                            ]
                        for l in range(4):
                            #                            key_pairwise_probabilistic_predictions = "%i_%i_%i_%i" % (i, j, n,l)
                            pairwise_probabilistic_predictions[f"{i}_{j}_{n}_{l}"] = (
                                current_pairwise_probabilistic_predictions_ij[l]
                            )
        elif (
            self.preference_order
            == PreferenceOrder.PARTIAL_ORDER
            # or self.preference_order == PreferenceOrder.BIPARTITE_PARTIAL_ORDER
        ):
            pairwise_probabilistic_predictions = {}
            for i in range(n_labels - 1):
                key_classifier_i = f"{i}"
                original_probabilistic_predictions_i = self.pairwise_classifier[
                    key_classifier_i
                ].predict_proba(X)
                presented_classes = list(
                    self.pairwise_classifier[key_classifier_i].classes_()
                )
                probabilistic_predictions_i = np.zeros((n_test_instances, 2))
                for c in range(2):
                    if c in presented_classes:
                        probabilistic_predictions_i[:, c] = (
                            original_probabilistic_predictions_i[
                                :, presented_classes.index(c)
                            ]
                        )
                for j in range(i + 1, n_labels):
                    key_classifier_j = f"{j}"
                    original_probabilistic_predictions_j = self.pairwise_classifier[
                        key_classifier_j
                    ].predict_proba(X)
                    presented_classes = list(
                        self.pairwise_classifier[key_classifier_j].classes_()
                    )
                    probabilistic_predictions_j = np.zeros((n_test_instances, 2))
                    for c in range(2):
                        if c in presented_classes:
                            probabilistic_predictions_j[:, c] = (
                                original_probabilistic_predictions_j[
                                    :, presented_classes.index(c)
                                ]
                            )
                    for n in range(n_test_instances):

                        current_pairwise_probabilistic_predictions_ij = [
                            probabilistic_predictions_i[n, 1]
                            * probabilistic_predictions_j[n, 0],
                            probabilistic_predictions_i[n, 0]
                            * probabilistic_predictions_j[n, 1],
                            probabilistic_predictions_i[n, 0]
                            * probabilistic_predictions_j[n, 0]
                            + probabilistic_predictions_i[n, 1]
                            * probabilistic_predictions_j[n, 1],
                        ]

                        # add a small regularization term if the probabilistic prediction is deterministic instead of probabilistic

                        if max(current_pairwise_probabilistic_predictions_ij) == 1:
                            current_pairwise_probabilistic_predictions_ij = [
                                x - 10**-10 if x == 1 else (10**-10) / 2
                                for x in current_pairwise_probabilistic_predictions_ij
                            ]
                        if min(current_pairwise_probabilistic_predictions_ij) == 0:
                            zero_indices = [
                                ind
                                for ind in range(3)
                                if current_pairwise_probabilistic_predictions_ij[ind]
                                == 0
                            ]
                            current_pairwise_probabilistic_predictions_ij = [
                                (
                                    (10**-10) / len(zero_indices)
                                    if x == 0
                                    else x - (10**-10) / (3 - len(zero_indices))
                                )
                                for x in current_pairwise_probabilistic_predictions_ij
                            ]
                        for l in range(3):
                            #                            key_pairwise_probabilistic_predictions = "%i_%i_%i_%i" % (i, j, n,l)
                            pairwise_probabilistic_predictions[f"{i}_{j}_{n}_{l}"] = (
                                current_pairwise_probabilistic_predictions_ij[l]
                            )
        return pairwise_probabilistic_predictions

    def predict_CLR(self, X, n_labels):
        n_instances, _ = X.shape
        calibrated_scores = np.zeros((n_instances))

        # For each label, get the calibrated score
        for k in range(n_labels):
            clf = self.calibrated_classifier[k]
            probabilistic_predictions = clf.predict_proba(X)
            _, n_classes = (
                probabilistic_predictions.shape
            )  # shape: [n_instances, n_classes]
            # print("n_classes _CLR", n_classes)
            if n_classes == 1:
                # TODO: debug this line
                # use any instance to find the predicted class
                if self.base_classifier.name in [
                    BaseLearnerName.XGBoost,
                    BaseLearnerName.LightGBM,
                ]:
                    pass
                # Check this label existing in the training set
                #
                else:
                    predicted_class = clf.predict(X[:2])
                    if predicted_class[0] == 1:
                        calibrated_scores += probabilistic_predictions

            else:
                # use probability at index 1
                calibrated_scores += probabilistic_predictions[:, 1]

        voting_scores = np.zeros((n_labels, n_instances))
        for k_1 in range(n_labels - 1):
            for k_2 in range(k_1 + 1, n_labels):  # TODO: check this line
                clf = self.pairwise_classifier[f"{k_1}_{k_2}"]
                probabilistic_predictions = clf.predict_proba(X)
                _, n_classes = probabilistic_predictions.shape
                if n_classes == 1:  # why 1? -> label at index 0?
                    if self.base_classifier.name in [
                        BaseLearnerName.XGBoost,
                        BaseLearnerName.LightGBM,
                    ]:
                        # Check this label existing in the training set
                        if self.single_label_pair[f"{k_1}_{k_2}"] == 0:
                            voting_scores[k_1, :] += [1 for n in range(n_instances)]
                        else:  # None will be handle in below with n_classes > 1
                            voting_scores[k_2, :] += [1 for n in range(n_instances)]
                    else:
                        predicted_class = clf.predict(
                            X[:2]
                        )  # :2 means first 2 instances
                        if predicted_class[0] == 0:
                            voting_scores[k_1, :] += [1 for n in range(n_instances)]
                        else:
                            voting_scores[k_2, :] += [1 for n in range(n_instances)]
                else:  # for classes > 1
                    voting_scores[k_1, :] += probabilistic_predictions[:, 0]
                    voting_scores[k_2, :] += probabilistic_predictions[:, 1]

        predicted_Y = []
        predicted_ranks = []
        for index in range(n_instances):
            prediction = [
                1 if voting_scores[k, index] >= calibrated_scores[index] else 0
                for k in range(n_labels)
            ]
            rank = [
                n_labels - sorted(voting_scores[:, index]).index(x)
                for x in voting_scores[:, index]
            ]
            predicted_Y.append(prediction)
            predicted_ranks.append(rank)

        return predicted_Y, predicted_ranks

    def fit(self, X, Y):
        """Training the model for each pair of labels (i, j), which could be pre-order or partial-order

        Args:
            X (_type_): _description_
            Y (_type_): _description_

        Raises:
            ValueError: _description_
        """
        log(INFO, f"Fitting model for preference order: {self.preference_order}")
        if "PRE_ORDER" in self.preference_order.name:
            self.pairwise_classifier = (
                self.base_classifier.pairwise_pre_order_classifier_fit(X, Y)
            )
        elif "PARTIAL_ORDER" in self.preference_order.name:
            self.pairwise_classifier = (
                self.base_classifier.pairwise_partial_order_classifier_fit(X, Y)
            )  # type: ignore
            pass
        else:
            raise ValueError(f"Unknown preference order: {self.preference_order}")

    def fit_CLR(self, X, Y):
        self.pairwise_classifier, self.calibrated_classifier, self.single_label_pair = (  # type: ignore
            self.base_classifier.pairwise_calibrated_classifier(X, Y)
        )

    def fit_BR(self, X, Y):
        """Train Binary Relevance model using scikit-learn's MultiOutputClassifier

        Args:
            X: Training features
            Y: Training labels
        """
        log(INFO, "Training Binary Relevance model")
        n_labels = Y.shape[1]

        # Create base classifier
        base_clf = self.base_classifier.get_classifier()  # type: ignore

        # Create MultiOutputClassifier
        self.br_classifier = MultiOutputClassifier(base_clf)

        # Train the model
        self.br_classifier.fit(X, Y)

        log(INFO, "Binary Relevance training completed")

    def predict_BR(self, X, n_labels):
        """Predict using Binary Relevance

        Args:
            X: Test features
            n_labels: Number of labels

        Returns:
            tuple: (predicted_Y, None) to match CLR interface
        """
        # Get predictions from MultiOutputClassifier and convert to list
        predicted_Y = self.br_classifier.predict(X)
        return np.array(predicted_Y).tolist(), None

    def fit_CC(self, X, Y):
        """Train Classifier Chain model using scikit-learn's ClassifierChain

        Args:
            X: Training features
            Y: Training labels
        """
        log(INFO, "Training Classifier Chain model")
        n_labels = Y.shape[1]

        # Create base classifier
        base_clf = self.base_classifier.get_classifier()  # type: ignore

        # Create ClassifierChain with random order
        self.cc_classifier = ClassifierChain(
            base_clf, order="random", random_state=RANDOM_STATE  # type: ignore
        )
        # Train the model
        self.cc_classifier.fit(X, Y)

        # Store the order for reference
        self.cc_order = self.cc_classifier.order_

        log(INFO, "Classifier Chain training completed")

    def predict_CC(self, X, n_labels):
        """Predict using Classifier Chain

        Args:
            X: Test features
            n_labels: Number of labels

        Returns:
            tuple: (predicted_Y, None) to match CLR interface
        """
        # Get predictions from ClassifierChain and convert to list
        predicted_Y = self.cc_classifier.predict(X)
        return np.array(predicted_Y).tolist(), None
