# -BaseClassifiers- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:02:21 2023

@author: nguyenli_admin
"""

import lightgbm
import numpy as np
from sklearn.base import BaseEstimator
from estimator import Estimator
from logging import INFO, log
from typing import Dict, List, Tuple, Optional
from numpy.typing import NDArray


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

        # Store all the pairwise classifiers for each pair of labels
        self.pairwise_classifiers: dict[str, Estimator] = {}

    # We may use dictionaries to store all the pairwise classifiers
    #   classifier = {}
    #    for i in range(n_labels - 1):
    #        for j in range(i + 1, n_labels):
    #            key = "%i_%i" % (i, j)
    #            classifier[key] = the pairwise_classifier for the label pair (y_i,y_j)

    def pairwise_calibrated_classifier(
        self, X: NDArray[np.float64], Y: NDArray[np.int32]
    ) -> Tuple[Dict[str, BaseEstimator], List[BaseEstimator]]:
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
        # With each label
        for k in range(n_labels):
            MCC_y = Y[:, k]
            # ex: k = 0 ->  Y[:,k] = [1, 0, 1, 0] --> MCC_y = [0, 1, 0, 1]
            # If the label is 1, then the MCC_y is 0
            MCC_y = np.logical_not(MCC_y).astype(int)

            # Learning for each label k
            classifier = Estimator(self.name)
            classifier.fit(X, MCC_y)
            calibrated_classifiers.append(classifier)

        pairwise_classifiers = {}
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
                # TODO: check this
                pairwise_classifiers[key] = self.base_learner.fit(MCC_X, MCC_y)  # type: ignore

        return pairwise_classifiers, calibrated_classifiers

    def pairwise_partial_order_classifier_fit(self, X, Y):
        # This BaseClassifier provides pairwise_probability_information for learning partial orders
        n_labels = len(Y[0])
        n_instances, _ = Y.shape
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
                # TODO: check this
                classifier = Estimator(self.name)
                classifier.fit(X, MCC_y)  # type: ignore
                pairwise_classifiers[key] = classifier
        return pairwise_classifiers
        # classifiers = []
        # for k_1 in range(n_labels - 1):
        #     local_classifier = []
        #     for k_2 in range(k_1 + 1, n_labels):
        #         MCC_y = []
        #         for n in range(n_instances):
        #             if Y[n, k_1] == Y[n, k_2]:
        #                 MCC_y.append(2)
        #             elif Y[n, k_1] == 1:
        #                 MCC_y.append(0)
        #             elif Y[n, k_2] == 1:
        #                 MCC_y.append(1)
        #         local_classifier.append(self.[base_learner]._3classifier(X, MCC_y))
        #     classifiers.append(local_classifier)

        # return classifiers

    def pairwise_pre_order_classifier_fit(self, X, Y) -> dict[str, Estimator]:
        """
        This BaseClassifier provides pairwise_probability_information for learning preorders
        For each pair of labels, we will train a classifier to predict the probability of the label
        """
        n_instances, n_labels = Y.shape
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

                # Init classifier
                classifier = Estimator(self.name)
                classifier.fit(X, MCC_y)  # type: ignore
                # Store the classifier for each pair of labels
                self.pairwise_classifiers[key] = classifier

        # This is a dictionary of pairwise classifiers. [key] is a string of the form "i_j"
        # where i and j are the indices of the labels in the label matrix Y
        return self.pairwise_classifiers
        # TODO: Do this `fit` trained or not? should return value or a classifier?

    def binary_relevance_classifer(self, X, Y) -> list[BaseEstimator]:
        # This is to learn a Binary relevance (BR)
        classifiers = []
        _, n_labels = Y.shape
        for k in range(n_labels):
            classifier = Estimator(self.name)
            classifier.fit(X, Y[:, k])
            classifiers.append(classifier)
        return classifiers
