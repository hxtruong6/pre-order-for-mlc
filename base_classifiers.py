"""Pairwise- and calibrated-classifier factories.

:class:`BaseClassifiers` builds the per-pair training datasets that
:class:`inference_models.PredictBOPOs` consumes: pairwise classifiers for
PRE_ORDER and PARTIAL_ORDER variants, and one calibrated classifier per
label for CLR. Estimator construction is delegated to
:class:`estimator.Estimator`, with fitting parallelised via joblib.
"""

from logging import INFO, log

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from estimator import Estimator, train_classifier


class BaseClassifiers:
    """Base classifiers for multi-label classification with different order types.

    Attributes:
        base_learner (Estimator): The base learning algorithm
        logger (logging.Logger): Logger instance
    """

    def __init__(self, name: str):
        log(
            INFO,
            f"BaseClassifiers: Initializing base learner: {name}",
        )
        self.name = name

    def get_classifier(self) -> BaseEstimator:
        return Estimator(self.name).get_classifier()  # type: ignore

    def pairwise_calibrated_classifier(self, X: NDArray[np.float64], Y: NDArray[np.int32]):
        """Train pairwise calibrated classifiers.

        Args:
            X: Input features of shape (n_samples, n_features)
            Y: Binary label matrix of shape (n_samples, n_labels)

            X_train = [[1, 2, 3, 0], [3, 2, 4, 1], [4, 5, 3, 3], [7, 6, 3, 1]]
            Y_train = [[1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]
            n_labels = 3

        Returns:
            Tuple containing:
                - Dictionary of pairwise classifiers
                - List of calibrated classifiers
        """
        n_instances, n_labels = Y.shape

        # calibrated_classifiers is in fact is a (inverse) BR classifier
        calibrated_classifiers = []
        clr_dataset_classifier = {}
        for k in range(n_labels):
            # MCC = score for class 0; invert the label column so the trained
            # classifier produces P(y_k = 0) directly (CLR convention).
            # Example: Y[:, k] = [1, 0, 1, 0] -> MCC_y = [0, 1, 0, 1].
            MCC_y = np.logical_not(Y[:, k]).astype(int)
            clr_dataset_classifier[str(k)] = {  # type: ignore
                "X": X.copy(),
                "Y": MCC_y.copy(),
            }

        log(
            INFO,
            f"\t - Training for {len(clr_dataset_classifier.keys())} calibrated_classifiers with {self.name}",
        )
        # run parraellly fit for each pair of labels
        classifiers = Parallel(n_jobs=-1)(
            delayed(train_classifier)(
                clr_dataset_classifier[str(k)]["X"],
                clr_dataset_classifier[str(k)]["Y"],
                self.name,
            )  # type: ignore
            for k in range(n_labels)
        )
        log(INFO, f"\t - Trained {n_labels} classifiers")

        calibrated_classifiers = list(classifiers)

        pairwise_classifiers = {}
        single_label_pair = {}
        dataset_classifier = {}
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                key = f"{i}_{j}"
                MCC_X = []
                MCC_y = []

                for n in range(n_instances):
                    if Y[n, i] == 1 and Y[n, j] == 0:
                        MCC_X.append(X[n])
                        MCC_y.append(0)
                    elif Y[n, i] == 0 and Y[n, j] == 1:
                        MCC_X.append(X[n])
                        MCC_y.append(1)

                dataset_classifier[key] = {  # type: ignore
                    "X": MCC_X,
                    "Y": MCC_y,
                }

        log(
            INFO,
            f"\t - Training for {len(dataset_classifier.keys())} pairs with {self.name}",
        )

        classifiers = Parallel(n_jobs=-1)(
            delayed(train_classifier)(
                dataset_classifier[key]["X"],
                dataset_classifier[key]["Y"],
                self.name,
            )  # type: ignore
            for key in dataset_classifier.keys()
        )

        log(INFO, f"\t - Trained {len(dataset_classifier.keys())} classifiers")

        pairwise_classifiers = dict(zip(dataset_classifier.keys(), classifiers))  # type: ignore

        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                key = f"{i}_{j}"
                MCC_y = dataset_classifier[key]["Y"]
                single_label_pair[key] = (
                    1
                    if len(np.unique(MCC_y)) == 1 and np.unique(MCC_y)[0] == 1
                    else (0 if len(np.unique(MCC_y)) == 1 and np.unique(MCC_y)[0] == 0 else None)
                )

        return pairwise_classifiers, calibrated_classifiers, single_label_pair

    def pairwise_partial_order_classifier_fit(self, X, Y):
        # This BaseClassifier provides pairwise_probability_information for learning partial orders
        n_labels = len(Y[0])
        n_instances, _ = Y.shape
        dataset_classifier = {}
        pairwise_classifiers = {}

        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                key = f"{i}_{j}"
                MCC_y = []
                for n in range(n_instances):
                    if Y[n, i] == Y[n, j]:
                        MCC_y.append(2)
                    elif Y[n, i] == 1 and Y[n, j] == 0:
                        MCC_y.append(0)
                    elif Y[n, i] == 0 and Y[n, j] == 1:
                        MCC_y.append(1)
                dataset_classifier[key] = {  # type: ignore
                    "X": X.copy(),
                    "Y": MCC_y.copy(),
                }
        log(
            INFO,
            f"\t - Training for {len(dataset_classifier.keys())} pairs with {self.name}",
        )
        # run parraellly fit for each pair of labels
        classifiers = Parallel(n_jobs=-1)(
            delayed(train_classifier)(
                dataset_classifier[key]["X"],
                dataset_classifier[key]["Y"],
                self.name,
            )  # type: ignore
            for key in dataset_classifier.keys()
        )
        log(INFO, f"\t - Trained {len(dataset_classifier.keys())} classifiers")

        pairwise_classifiers = dict(zip(dataset_classifier.keys(), classifiers))  # type: ignore

        return pairwise_classifiers  # type: ignore

    def pairwise_pre_order_classifier_fit(self, X, Y) -> dict[str, Estimator]:
        """
        This BaseClassifier provides pairwise_probability_information for learning preorders
        For each pair of labels, we will train a classifier to predict the probability of the label
        """
        n_instances, n_labels = Y.shape
        log(INFO, f"\t - {n_labels} labels with {self.name}")
        dataset_classifier = {}
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                key = f"{i}_{j}"
                MCC_y = []
                for n in range(n_instances):
                    if Y[n, i] == 0 and Y[n, j] == 0:
                        MCC_y.append(2)
                    elif Y[n, i] == 1 and Y[n, j] == 1:
                        MCC_y.append(3)
                    elif Y[n, i] == 1 and Y[n, j] == 0:
                        MCC_y.append(0)
                    elif Y[n, i] == 0 and Y[n, j] == 1:
                        MCC_y.append(1)

                dataset_classifier[key] = {  # type: ignore
                    "X": X.copy(),
                    "Y": MCC_y.copy(),
                }
        log(
            INFO,
            f"\t - Training {len(dataset_classifier.keys())} pairs with {self.name}",
        )
        # run parraellly fit for each pair of labels
        classifiers = Parallel(n_jobs=-1)(
            delayed(train_classifier)(
                dataset_classifier[key]["X"],
                dataset_classifier[key]["Y"],
                self.name,
            )  # type: ignore
            for key in dataset_classifier.keys()
        )
        log(INFO, f"\t - Trained {len(dataset_classifier.keys())} classifiers")

        pairwise_classifiers = dict(zip(dataset_classifier.keys(), classifiers))  # type: ignore

        # This is a dictionary of pairwise classifiers. [key] is a string of the form "i_j"
        # where i and j are the indices of the labels in the label matrix Y
        return pairwise_classifiers  # type: ignore
