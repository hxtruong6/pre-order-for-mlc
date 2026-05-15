"""Inference models for BOPOs and the comparison baselines (CLR, BR, CC).

:class:`PredictBOPOs` is the central model. ``fit`` / ``predict_proba``
train the pairwise calibrated classifiers required by both pre- and
partial-order BOPOs and emit the marginal scores used for ranking
metrics. ``fit_CLR`` / ``fit_BR`` / ``fit_CC`` (and their predict
counterparts) train the published baselines on the same splits so all
algorithms share data, seeds, and base learners.

``predict_preference_orders`` then turns the pairwise probabilities into
a per-instance ordered prediction by dispatching to the ILP search in
:mod:`searching_algorithms` for one of the eight ``(target_metric,
height)`` combinations enumerated as IA1-IA8 in :mod:`evaluation_test`.
"""

from enum import Enum
from logging import INFO, log

import numpy as np
from joblib import Parallel, delayed
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

from base_classifiers import BaseClassifiers
from constants import RANDOM_STATE, BaseLearnerName, TargetMetric
from estimator import Estimator
from searching_algorithms import Search_BOParOs, Search_BOPreOs


class PreferenceOrder(Enum):
    """Variant of the underlying preference order learned by BOPOs."""

    PRE_ORDER = "PreOrder"
    PARTIAL_ORDER = "PartialOrder"


class PredictBOPOs:
    """Pairwise-classifier-based multi-label predictor.

    Trains :math:`K(K-1)/2` pairwise probabilistic classifiers (PRE_ORDER:
    four classes per pair; PARTIAL_ORDER: three classes per pair), an
    auxiliary binary-relevance head for marginal probabilities, and
    optionally CLR / BR / CC baselines that share the same base learner.
    """

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

        # Auxiliary BR head used to derive per-label marginal probabilities for
        # BOPOs (both PRE_ORDER and PARTIAL_ORDER). Set in fit().
        self.br_head_for_proba: MultiOutputClassifier | None = None

    def predict_preference_orders(
        self,
        pairwise_probabilistic_predictions,
        n_labels,
        n_instances,
        target_metric: TargetMetric,
        height: int | None = None,
    ) -> tuple[list[int], list[float], list[int] | None, list[int] | None]:
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
                (
                    predict_BOPOS,
                    predict_binary_vectors,
                    indices_vector,
                    prediction_with_partial_abstention,
                ) = search_BOPrerOs.PRE_ORDER()
            elif self.preference_order == PreferenceOrder.PARTIAL_ORDER:
                (
                    predict_BOPOS,
                    predict_binary_vectors,
                    indices_vector,
                    prediction_with_partial_abstention,
                ) = search_BOParOs.PARTIAL_ORDER()
            else:
                raise ValueError(f"[Hamming] Unknown preference order: {self.preference_order}")

        elif target_metric == TargetMetric.Subset:
            if self.preference_order == PreferenceOrder.PRE_ORDER:
                (
                    predict_BOPOS,
                    predict_binary_vectors,
                    indices_vector,
                    prediction_with_partial_abstention,
                ) = search_BOPrerOs.PRE_ORDER()
            elif self.preference_order == PreferenceOrder.PARTIAL_ORDER:
                (
                    predict_BOPOS,
                    predict_binary_vectors,
                    indices_vector,
                    prediction_with_partial_abstention,
                ) = search_BOParOs.PARTIAL_ORDER()

            else:
                raise ValueError(f"[Subset] Unknown preference order: {self.preference_order}")

        return (
            predict_BOPOS,
            predict_binary_vectors,
            indices_vector,  # type: ignore
            prediction_with_partial_abstention,
        )  # type: ignore

    def predict_proba(self, X, n_labels):
        n_test_instances, _ = X.shape
        if self.preference_order == PreferenceOrder.PRE_ORDER:

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
                    pairwise_probabilistic_predictions_ij = np.zeros((n_test_instances, 4))
                    key_classifier = f"{i}_{j}"

                    original_pairwise_probabilistic_predictions_ij = pairwise_predict_proba[
                        key_classifier
                    ]
                    # The pairwise classifier may have seen fewer than 4 (PRE_ORDER)
                    # or 3 (PARTIAL_ORDER) classes in training; align the columns
                    # below by inserting zeros for any missing class.
                    presented_classes = list(self.pairwise_classifier[key_classifier].classes_())
                    for l in range(4):
                        if l in presented_classes:
                            pairwise_probabilistic_predictions_ij[:, l] = (
                                original_pairwise_probabilistic_predictions_ij[
                                    :, presented_classes.index(l)
                                ]
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
                                if current_pairwise_probabilistic_predictions_ij[ind] == 0
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
        elif self.preference_order == PreferenceOrder.PARTIAL_ORDER:

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
                    pairwise_probabilistic_predictions_ij = np.zeros((n_test_instances, 3))
                    key_classifier = f"{i}_{j}"
                    original_pairwise_probabilistic_predictions_ij = pairwise_predict_proba[
                        key_classifier
                    ]
                    presented_classes = list(self.pairwise_classifier[key_classifier].classes_())
                    for l in range(3):
                        if l in presented_classes:
                            pairwise_probabilistic_predictions_ij[:, l] = (
                                original_pairwise_probabilistic_predictions_ij[
                                    :, presented_classes.index(l)
                                ]
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
                                if current_pairwise_probabilistic_predictions_ij[ind] == 0
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
                            pairwise_probabilistic_predictions[f"{i}_{j}_{n}_{l}"] = (
                                current_pairwise_probabilistic_predictions_ij[l]
                            )
        return pairwise_probabilistic_predictions

    def predict_proba_BR(self, X, n_labels):
        n_test_instances, _ = X.shape
        if self.preference_order == PreferenceOrder.PRE_ORDER:
            pairwise_probabilistic_predictions = {}
            for i in range(n_labels - 1):
                key_classifier_i = f"{i}"
                original_probabilistic_predictions_i = self.pairwise_classifier[
                    key_classifier_i
                ].predict_proba(X)
                presented_classes = list(self.pairwise_classifier[key_classifier_i].classes_())
                probabilistic_predictions_i = np.zeros((n_test_instances, 2))
                for c in range(2):
                    if c in presented_classes:
                        probabilistic_predictions_i[:, c] = original_probabilistic_predictions_i[
                            :, presented_classes.index(c)
                        ]
                for j in range(i + 1, n_labels):
                    key_classifier_j = f"{j}"
                    original_probabilistic_predictions_j = self.pairwise_classifier[
                        key_classifier_j
                    ].predict_proba(X)
                    presented_classes = list(self.pairwise_classifier[key_classifier_j].classes_())
                    probabilistic_predictions_j = np.zeros((n_test_instances, 2))
                    for c in range(2):
                        if c in presented_classes:
                            probabilistic_predictions_j[:, c] = (
                                original_probabilistic_predictions_j[:, presented_classes.index(c)]
                            )
                    for n in range(n_test_instances):

                        current_pairwise_probabilistic_predictions_ij = [
                            probabilistic_predictions_i[n, 1] * probabilistic_predictions_j[n, 0],
                            probabilistic_predictions_i[n, 0] * probabilistic_predictions_j[n, 1],
                            probabilistic_predictions_i[n, 0] * probabilistic_predictions_j[n, 0],
                            probabilistic_predictions_i[n, 1] * probabilistic_predictions_j[n, 1],
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
                                if current_pairwise_probabilistic_predictions_ij[ind] == 0
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
        elif self.preference_order == PreferenceOrder.PARTIAL_ORDER:
            pairwise_probabilistic_predictions = {}
            for i in range(n_labels - 1):
                key_classifier_i = f"{i}"
                original_probabilistic_predictions_i = self.pairwise_classifier[
                    key_classifier_i
                ].predict_proba(X)
                presented_classes = list(self.pairwise_classifier[key_classifier_i].classes_())
                probabilistic_predictions_i = np.zeros((n_test_instances, 2))
                for c in range(2):
                    if c in presented_classes:
                        probabilistic_predictions_i[:, c] = original_probabilistic_predictions_i[
                            :, presented_classes.index(c)
                        ]
                for j in range(i + 1, n_labels):
                    key_classifier_j = f"{j}"
                    original_probabilistic_predictions_j = self.pairwise_classifier[
                        key_classifier_j
                    ].predict_proba(X)
                    presented_classes = list(self.pairwise_classifier[key_classifier_j].classes_())
                    probabilistic_predictions_j = np.zeros((n_test_instances, 2))
                    for c in range(2):
                        if c in presented_classes:
                            probabilistic_predictions_j[:, c] = (
                                original_probabilistic_predictions_j[:, presented_classes.index(c)]
                            )
                    for n in range(n_test_instances):

                        current_pairwise_probabilistic_predictions_ij = [
                            probabilistic_predictions_i[n, 1] * probabilistic_predictions_j[n, 0],
                            probabilistic_predictions_i[n, 0] * probabilistic_predictions_j[n, 1],
                            probabilistic_predictions_i[n, 0] * probabilistic_predictions_j[n, 0]
                            + probabilistic_predictions_i[n, 1] * probabilistic_predictions_j[n, 1],
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
                                if current_pairwise_probabilistic_predictions_ij[ind] == 0
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
        calibrated_scores = np.zeros(n_instances)

        # For each label, get the calibrated score
        for k in range(n_labels):
            clf = self.calibrated_classifier[k]
            probabilistic_predictions = clf.predict_proba(X)
            _, n_classes = probabilistic_predictions.shape
            if n_classes == 1:
                # Degenerate fold: the calibrated classifier saw only one class.
                # For tree boosters we cannot safely interpret the column, so
                # skip; otherwise probe one instance to decide whether the
                # surviving class is the positive label.
                if self.base_classifier.name in [
                    BaseLearnerName.XGBoost,
                    BaseLearnerName.LightGBM,
                ]:
                    pass
                else:
                    predicted_class = clf.predict(X[:2])
                    if predicted_class[0] == 1:
                        calibrated_scores += probabilistic_predictions
            else:
                calibrated_scores += probabilistic_predictions[:, 1]

        voting_scores = np.zeros((n_labels, n_instances))
        for k_1 in range(n_labels - 1):
            for k_2 in range(k_1 + 1, n_labels):
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
                        predicted_class = clf.predict(X[:2])  # :2 means first 2 instances
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
                n_labels - sorted(voting_scores[:, index]).index(x) for x in voting_scores[:, index]
            ]
            predicted_Y.append(prediction)
            predicted_ranks.append(rank)

        # Marginal-like score: vote-fraction over pairwise comparisons in [0, 1].
        # voting_scores has shape (n_labels, n_instances). Transpose to (n, K).
        Y_proba = voting_scores.T / max(1, n_labels - 1)
        Y_proba = np.clip(Y_proba, 0.0, 1.0)

        return predicted_Y, predicted_ranks, Y_proba

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
            self.pairwise_classifier = self.base_classifier.pairwise_pre_order_classifier_fit(X, Y)
        elif "PARTIAL_ORDER" in self.preference_order.name:
            self.pairwise_classifier = self.base_classifier.pairwise_partial_order_classifier_fit(
                X, Y
            )  # type: ignore
            pass
        else:
            raise ValueError(f"Unknown preference order: {self.preference_order}")

        # Auxiliary BR head to produce per-label marginal probabilities for
        # downstream ranking metrics. We train this for both PRE_ORDER and
        # PARTIAL_ORDER so the marginal-derivation rule is identical across
        # BOPOs variants.
        try:
            base_clf = self.base_classifier.get_classifier()  # type: ignore
            self.br_head_for_proba = MultiOutputClassifier(base_clf)
            self.br_head_for_proba.fit(X, Y)
            log(INFO, "Auxiliary BR head for marginal proba trained")
        except Exception as e:  # pragma: no cover - defensive
            log(INFO, f"Auxiliary BR head training failed: {e}")
            self.br_head_for_proba = None

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
        Y.shape[1]

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
            tuple: (predicted_Y, None, Y_proba) where Y_proba is (n, K) float
            in [0, 1] with P(y_k = 1 | x).
        """
        # Get predictions from MultiOutputClassifier and convert to list
        predicted_Y = self.br_classifier.predict(X)

        # Per-label marginal probability of the positive class.
        proba_list = self.br_classifier.predict_proba(X)
        Y_proba = _stack_positive_class_proba(self.br_classifier.estimators_, proba_list)
        return np.array(predicted_Y).tolist(), None, Y_proba

    def fit_CC(self, X, Y):
        """Train Classifier Chain model using scikit-learn's ClassifierChain

        Args:
            X: Training features
            Y: Training labels
        """
        log(INFO, "Training Classifier Chain model")
        Y.shape[1]

        # Create base classifier
        base_clf = self.base_classifier.get_classifier()  # type: ignore

        # Create ClassifierChain with random order
        self.cc_classifier = ClassifierChain(
            base_clf, order=None, random_state=RANDOM_STATE  # type: ignore
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
            tuple: (predicted_Y, None, Y_proba) where Y_proba is the (n, K)
            per-label probability matrix returned by ClassifierChain.
        """
        # Get predictions from ClassifierChain and convert to list
        predicted_Y = self.cc_classifier.predict(X)
        # ClassifierChain.predict_proba already returns shape (n, K).
        Y_proba = np.asarray(self.cc_classifier.predict_proba(X), dtype=float)
        Y_proba = np.clip(Y_proba, 0.0, 1.0)
        return np.array(predicted_Y).tolist(), None, Y_proba

    def predict_marginal_proba(self, X) -> np.ndarray:
        """Return per-label marginal probabilities from the auxiliary BR head.

        Output shape: (n_instances, n_labels), values in [0, 1].
        """
        if self.br_head_for_proba is None:
            raise RuntimeError("predict_marginal_proba called before BR head was trained")
        proba_list = self.br_head_for_proba.predict_proba(X)
        Y_proba = _stack_positive_class_proba(self.br_head_for_proba.estimators_, proba_list)
        return Y_proba


def _stack_positive_class_proba(estimators, proba_list) -> np.ndarray:
    """Stack per-label probability of the positive class (label==1).

    Each estimator's predict_proba returns shape (n, n_classes_k). For binary
    labels this is (n, 2) with columns [P(y=0), P(y=1)]. For constant-label
    folds it can be (n, 1) with a single class; in that case we emit the
    appropriate constant column based on which class is present.
    """
    cols = []
    for k, p in enumerate(proba_list):
        p = np.asarray(p, dtype=float)
        classes = getattr(estimators[k], "classes_", None)
        if p.ndim == 2 and p.shape[1] == 2:
            # Standard binary case; column index of class==1.
            if classes is not None:
                try:
                    idx_pos = list(classes).index(1)
                except ValueError:
                    idx_pos = 1
            else:
                idx_pos = 1
            cols.append(p[:, idx_pos])
        elif p.ndim == 2 and p.shape[1] == 1:
            # Single-class fold: emit 1 if the present class is the positive
            # one, else 0. The probability column is uninformative (all ones).
            if classes is not None and 1 in list(classes):
                cols.append(np.ones(p.shape[0], dtype=float))
            else:
                cols.append(np.zeros(p.shape[0], dtype=float))
        else:
            # Defensive fallback: collapse to a constant column.
            cols.append(np.zeros(p.shape[0], dtype=float))
    Y_proba = np.column_stack(cols)
    return np.clip(Y_proba, 0.0, 1.0)
