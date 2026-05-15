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

from preorder4mlc.base_classifiers import BaseClassifiers
from preorder4mlc.constants import RANDOM_STATE, BaseLearnerName, TargetMetric
from preorder4mlc.estimator import Estimator
from preorder4mlc.searching_algorithms import Search_BOParOs, Search_BOPreOs


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
        self.base_classifier: BaseClassifiers = BaseClassifiers(base_classifier_name)
        self.preference_order = preference_order

        # Populated by fit() (keys "i_j") and by fit_CLR() (keys "i_j" plus
        # the per-label calibrated_classifier list).
        self.pairwise_classifier: dict[str, Estimator] = {}
        self.calibrated_classifier: list[Estimator] = []
        self.single_label_pair: dict[str, int | None] = {}

        # BR / CC baselines. Set in fit_BR() / fit_CC(), read in predict_BR() /
        # predict_CC(). Order is preserved on cc_classifier.order_.
        self.br_classifier: MultiOutputClassifier | None = None
        self.cc_classifier: ClassifierChain | None = None
        self.cc_order: list[int] = []

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

        # target_metric is already baked into search_BOPrerOs / search_BOParOs
        # via their constructors above; the only remaining choice is which
        # variant to dispatch to.
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
            raise ValueError(f"Unknown preference order: {self.preference_order}")

        return (
            predict_BOPOS,
            predict_binary_vectors,
            indices_vector,
            prediction_with_partial_abstention,
        )

    def predict_proba(self, X, n_labels):
        n_test_instances, _ = X.shape
        n_classes = 4 if self.preference_order == PreferenceOrder.PRE_ORDER else 3

        pairwise_predict_proba = self._parallel_pairwise_predict_proba(X, n_labels)

        pairwise_probabilistic_predictions: dict[str, float] = {}
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                key_classifier = f"{i}_{j}"
                aligned = _align_to_n_classes(
                    classifier=self.pairwise_classifier[key_classifier],
                    raw_proba=pairwise_predict_proba[key_classifier],
                    n_test_instances=n_test_instances,
                    n_classes=n_classes,
                )
                for n in range(n_test_instances):
                    row = _regularize_proba_row(aligned[n], n_classes)
                    for l in range(n_classes):
                        pairwise_probabilistic_predictions[f"{i}_{j}_{n}_{l}"] = row[l]
        return pairwise_probabilistic_predictions

    def _parallel_pairwise_predict_proba(self, X, n_labels):
        """Run predict_proba on each "i_j" pairwise classifier in parallel.

        Returns a dict keyed by the pairwise key in the same order as
        self.pairwise_classifier (matches the (i, j) iteration order).
        """
        predict_results = Parallel(n_jobs=-1)(
            delayed(self.pairwise_classifier[f"{i}_{j}"].predict_proba)(X)
            for i in range(n_labels - 1)
            for j in range(i + 1, n_labels)
        )
        return dict(zip(self.pairwise_classifier.keys(), predict_results))

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


def _align_to_n_classes(classifier, raw_proba, n_test_instances, n_classes):
    """Project a pairwise classifier's predict_proba output into the fixed
    n_classes-column layout expected by the BOPOs encoders.

    A pairwise classifier may have seen fewer than n_classes labels in
    training (n_classes is 4 for PRE_ORDER, 3 for PARTIAL_ORDER). Missing
    classes are filled with zeros.
    """
    aligned = np.zeros((n_test_instances, n_classes))
    presented_classes = list(classifier.classes_())
    for l in range(n_classes):
        if l in presented_classes:
            aligned[:, l] = raw_proba[:, presented_classes.index(l)]
    return aligned


def _regularize_proba_row(probs, n_classes):
    """Nudge a single probability row away from the degenerate {0, 1} corners.

    Required because the downstream ILP expects strictly positive scores; a
    deterministic classifier output (max == 1 or min == 0) is shifted by
    ``1e-10`` redistributed across the remaining slots. Mirrors the
    pre-refactor logic in ``predict_proba`` exactly:

    1. If any entry equals 1, subtract 1e-10 from it and add 1e-10/(n-1)
       to each non-1 entry.
    2. Then, if any entry is still 0, give it 1e-10/k (k = number of
       zeros) and pay for it by subtracting 1e-10/(n-k) from each non-zero
       entry. The two passes can chain: step 2 inspects the row mutated
       by step 1.
    """
    if max(probs) == 1:
        probs = [x - 10**-10 if x == 1 else (10**-10) / (n_classes - 1) for x in probs]
    if min(probs) == 0:
        zero_indices = [ind for ind in range(n_classes) if probs[ind] == 0]
        probs = [
            (
                (10**-10) / len(zero_indices)
                if x == 0
                else x - (10**-10) / (n_classes - len(zero_indices))
            )
            for x in probs
        ]
    return probs


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
