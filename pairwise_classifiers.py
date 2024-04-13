# -pairwise_classifiers- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:02:21 2023

@author: nguyenli_admin
"""

from base_classifiers import BaseClassifiers


class PairwiseClassifiers:
    def __init__(self, base_learner_name: str):
        self.base_learner = BaseClassifiers(base_learner_name)

    # We may use dictionaries to store all the pairwise classifiers
    #   classifier = {}
    #    for i in range(n_labels - 1):
    #        for j in range(i + 1, n_labels):
    #            key = "%i_%i" % (i, j)
    #            classifier[key] = the pairwise_classifier for the label pair (y_i,y_j)

    def _pairwise_2classifier(self, n_labels, X, Y):
        n_instances, _ = Y.shape
        calibrated_classifiers = []
        for k in range(n_labels):
            MCC_y = []
            for n in range(n_instances):
                if Y[n, k] == 1:
                    MCC_y.append(0)
                else:
                    MCC_y.append(1)
            calibrated_classifiers.append(self.base_learner._2classifier(X, MCC_y))
        classifiers = []
        for k_1 in range(n_labels - 1):
            local_classifier = []
            for k_2 in range(k_1 + 1, n_labels):
                MCC_X = []
                MCC_y = []
                for n in range(n_instances):
                    if Y[n, k_1] == 1:
                        MCC_X.append(X[n])
                        MCC_y.append(0)
                    elif Y[n, k_2] == 1:
                        MCC_X.append(X[n])
                        MCC_y.append(1)

                local_classifier.append(self.base_learner._2classifier(MCC_X, MCC_y))
            classifiers.append(local_classifier)

        return classifiers, calibrated_classifiers

    def _pairwise_3classifier(self, n_labels, X, Y):
        n_instances, _ = Y.shape
        classifiers = []
        for k_1 in range(n_labels - 1):
            local_classifier = []
            for k_2 in range(k_1 + 1, n_labels):
                MCC_y = []
                for n in range(n_instances):
                    if Y[n, k_1] == Y[n, k_2]:
                        MCC_y.append(2)
                    elif Y[n, k_1] == 1:
                        MCC_y.append(0)
                    elif Y[n, k_2] == 1:
                        MCC_y.append(1)
                local_classifier.append(self.base_learner._3classifier(X, MCC_y))
            classifiers.append(local_classifier)

        return classifiers

    def _pairwise_4classifier(self, n_labels, X, Y):
        n_instances, _ = Y.shape
        classifiers = []
        for k_1 in range(n_labels - 1):
            local_classifier = []
            for k_2 in range(k_1 + 1, n_labels):
                MCC_y = []
                for n in range(n_instances):
                    if Y[n, k_1] == 0 and Y[n, k_2] == 0:
                        MCC_y.append(2)
                    elif Y[n, k_1] == 1 and Y[n, k_2] == 1:
                        MCC_y.append(3)
                    elif Y[n, k_1] == 1:
                        MCC_y.append(0)
                    elif Y[n, k_2] == 1:
                        MCC_y.append(1)
                local_classifier.append(self.base_learner._4classifier(X, MCC_y))
            classifiers.append(local_classifier)

        return classifiers

    def _BR(self, n_labels, X, Y):
        classifiers = []
        for k in range(n_labels):
            classifiers.append(self.base_learner._2classifier(X, Y[:, k]))
        return classifiers
