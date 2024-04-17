# -predictors_partial- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:44:16 2023

@author: nguyenli_admin
"""

import numpy as np
from scipy.io import arff
import pandas as pd
from scipy.stats import bernoulli
from sklearn.model_selection import KFold

from base_classifer import BaseClassifiers


class predictors:

    def __init__(self, n_labels, base_learner: str):
        self.base_learner = base_learner

    def save_model(model, filename):
        import pickle

        with open(filename, "wb") as f:
            pickle.dump(model, f)

    def _CLR(self, X_test, pairwise_2classifier, calibrated_2classifier):
        n_instances, _ = X_test.shape
        calibrated_scores = np.zeros((n_instances))
        for k in range(n_labels):
            clf = calibrated_2classifier[k]
            probabilistic_predictions = clf.predict_proba(X_test)
            _, n_classes = probabilistic_predictions.shape
            if n_classes == 1:
                predicted_class = clf.predict(X_test[:2])
                if predicted_class[0] == 1:
                    calibrated_scores += probabilistic_predictions
            else:
                calibrated_scores += probabilistic_predictions[:, 1]
        voting_scores = np.zeros((n_labels, n_instances))
        for k_1 in range(n_labels - 1):
            local_classifier = pairwise_2classifier[k_1]
            for k_2 in range(n_labels - k_1 - 1):
                clf = local_classifier[k_2]
                probabilistic_predictions = clf.predict_proba(X_test)
                _, n_classes = probabilistic_predictions.shape
                if n_classes == 1:
                    predicted_class = clf.predict(X_test[:2])
                    if predicted_class[0] == 0:
                        voting_scores[k_1, :] += [1 for n in range(n_instances)]
                    else:
                        voting_scores[k_1 + k_2 + 1, :] += [
                            1 for n in range(n_instances)
                        ]
                else:
                    voting_scores[k_1, :] += probabilistic_predictions[:, 0]
                    voting_scores[k_1 + k_2 + 1, :] += probabilistic_predictions[:, 1]
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

    def _partialorders(
        self,
        X_test,
        Y_test,
        known_indices,
        pairwise_3classifier,
        calibrated_2classifier,
    ):
        indices_vector = {}
        indVec = 0
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                for l in range(3):
                    key = "%i_%i_%i" % (i, j, l)
                    indices_vector[key] = indVec
                    indVec += 1
        G, h, A, b, I, B = predictors._partialorders_compute_parameters(
            base_learner, indices_vector, n_labels
        )
        hard_predictions = []
        predicted_preorders = []
        n_instances, _ = X_test.shape
        for ind in range(n_instances):
            #            print(index, n_instances)
            vector = []
            #            indexEmpty = []
            for i in range(n_labels - 1):
                local_classifier = pairwise_3classifier[i]
                for k_2 in range(n_labels - i - 1):
                    # #                for j in range(i+1,n_labels):
                    #                     j = i + k_2 + 1
                    #                     if i in known_indices[ind] and j in known_indices[ind]:
                    #                         if Y_test[ind,i] == Y_test[ind,j]:
                    #                             pairInfor = [0, 0, 1]
                    #                         elif Y_test[ind,i] > Y_test[ind,j]:
                    #                             pairInfor = [1, 0, 0]
                    #                         elif Y_test[ind,i] < Y_test[ind,j]:
                    #                             pairInfor = [0, 1, 0]
                    #                     else:
                    #                         clf = local_classifier[k_2]
                    #                         pairInfor_ori = clf.predict_proba(X_test[ind].reshape(1, -1))
                    #                         pairInfor = pairInfor_ori[0]
                    #                         presented_classes = list(clf.classes_)
                    #                         if len(presented_classes) < 3:
                    #                             pairInfor = [pairInfor[presented_classes.index(x)] if x in presented_classes else 0 for x in range(n_labels)]
                    #                         if i in known_indices[ind] and j not in known_indices[ind]:
                    #                             if Y_test[ind,i]  == 1:
                    #                                 pairInfor = [pairInfor[0], 0, pairInfor[2]]
                    #                                 pairInfor = [x/sum(pairInfor) for x in pairInfor]
                    #                             elif Y_test[ind,i]  == 0:
                    #                                 pairInfor = [0, pairInfor[1], pairInfor[2]]
                    #                                 pairInfor = [x/sum(pairInfor) for x in pairInfor]
                    #                         elif i not in known_indices[ind] and j in known_indices[ind]:
                    #                             if Y_test[ind,j]  == 1:
                    #                                 pairInfor = [0, pairInfor[1], pairInfor[2]]
                    #                                 pairInfor = [x/sum(pairInfor) for x in pairInfor]
                    #                             elif Y_test[ind,j]  == 0:
                    #                                 pairInfor = [pairInfor[0], 0, pairInfor[2]]
                    #                                 pairInfor = [x/sum(pairInfor) for x in pairInfor]
                    clf = local_classifier[k_2]
                    pairInfor_ori = clf.predict_proba(X_test[ind].reshape(1, -1))
                    pairInfor = pairInfor_ori[0]
                    presented_classes = list(clf.classes_)
                    if len(presented_classes) < 3:
                        pairInfor = [
                            (
                                pairInfor[presented_classes.index(x)]
                                if x in presented_classes
                                else 0
                            )
                            for x in range(n_labels)
                        ]

                    # add a small regularization term if the probabilistic prediction is deterministic instead of probabilistic
                    if max(pairInfor) == 1:
                        pairInfor = [
                            x - 10**-10 if x == 1 else (10**-10) / 2 for x in pairInfor
                        ]
                    if min(pairInfor) == 0:
                        zero_indices = [ind for ind in range(3) if pairInfor[ind] == 0]
                        pairInfor = [
                            (
                                (10**-10) / len(zero_indices)
                                if x == 0
                                else x - (10**-10) / (3 - len(zero_indices))
                            )
                            for x in pairInfor
                        ]
                    pairInfor = [-np.log(pairInfor[l]) for l in range(3)]
                    vector += pairInfor
            # Modify A to take into account the information of partial labes at the test time
            if len(known_indices[ind]) > 0:
                rowA = 0
                for i in range(n_labels - 1):
                    for j in range(i + 1, n_labels):
                        if i in known_indices[ind] and j in known_indices[ind]:
                            if Y_test[ind, i] == Y_test[ind, j]:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 1
                            elif Y_test[ind, i] > Y_test[ind, j]:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 0
                            elif Y_test[ind, i] < Y_test[ind, j]:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 0

                        elif i in known_indices[ind] and j not in known_indices[ind]:
                            if Y_test[ind, i] == 1:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 1
                            elif Y_test[ind, i] == 0:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 1

                        elif i not in known_indices[ind] and j in known_indices[ind]:
                            if Y_test[ind, j] == 1:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 1
                            elif Y_test[ind, j] == 0:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 1
                        rowA += 1

            Gtest = np.array(G)
            Atest = np.array(A)
            # for indCol in indexEmpty:
            #     Gtest[:, indCol] = 0
            #     Atest[:, indCol] = 0
            hard_prediction_indices, predicted_preorder = (
                predictors._partialorders_reasoning_procedure(
                    base_learner,
                    vector,
                    indices_vector,
                    n_labels,
                    Gtest,
                    h,
                    Atest,
                    b,
                    I,
                    B,
                )
            )
            hard_prediction = [
                1 if x in hard_prediction_indices else 0 for x in range(n_labels)
            ]
            hard_predictions.append(hard_prediction)
            predicted_preorders.append(predicted_preorder)
        return hard_predictions, predicted_preorders

    def _partialorders_compute_parameters(self, indices_vector, n_labels):
        G = np.zeros(
            (
                int(n_labels * (n_labels - 1) * (n_labels - 2)),
                int(n_labels * (n_labels - 1) * 1.5),
            )
        )
        rowG = 0
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                for k in range(i):
                    indVecs = [
                        indices_vector[val]
                        for val in [
                            "%i_%i_%i" % (i, j, 0),
                            "%i_%i_%i" % (k, i, 1),
                            "%i_%i_%i" % (k, j, 0),
                        ]
                    ]
                    for ind in range(1):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(1, 3):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                    indVecs = [
                        indices_vector[val]
                        for val in [
                            "%i_%i_%i" % (i, j, 1),
                            "%i_%i_%i" % (k, i, 0),
                            "%i_%i_%i" % (k, j, 1),
                        ]
                    ]
                    for ind in range(1):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(1, 3):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                for k in range(i + 1, j):
                    indVecs = [
                        indices_vector[val]
                        for val in [
                            "%i_%i_%i" % (i, j, 0),
                            "%i_%i_%i" % (i, k, 0),
                            "%i_%i_%i" % (k, j, 0),
                        ]
                    ]
                    for ind in range(1):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(1, 3):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                    indVecs = [
                        indices_vector[val]
                        for val in [
                            "%i_%i_%i" % (i, j, 1),
                            "%i_%i_%i" % (i, k, 1),
                            "%i_%i_%i" % (k, j, 1),
                        ]
                    ]
                    for ind in range(1):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(1, 3):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                for k in range(j + 1, n_labels):
                    indVecs = [
                        indices_vector[val]
                        for val in [
                            "%i_%i_%i" % (i, j, 0),
                            "%i_%i_%i" % (i, k, 0),
                            "%i_%i_%i" % (j, k, 1),
                        ]
                    ]
                    for ind in range(1):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(1, 3):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                    indVecs = [
                        indices_vector[val]
                        for val in [
                            "%i_%i_%i" % (i, j, 1),
                            "%i_%i_%i" % (i, k, 1),
                            "%i_%i_%i" % (j, k, 0),
                        ]
                    ]
                    for ind in range(1):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(1, 3):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
        h = np.ones((n_labels * (n_labels - 1) * (n_labels - 2), 1))
        A = np.zeros(
            (int(n_labels * (n_labels - 1) * 0.5), int(n_labels * (n_labels - 1) * 1.5))
        )
        rowA = 0
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                # we can inject the information of partial labels at test time here
                for l in range(3):
                    indVec = indices_vector["%i_%i_%i" % (i, j, l)]
                    A[rowA, indVec] = 1
                rowA += 1
        b = np.ones((int(n_labels * (n_labels - 1) * 0.5), 1))
        I = set()
        B = set(range(int(n_labels * (n_labels - 1) * 1.5)))
        return G, h, A, b, I, B

    def _partialorders_reasoning_procedure(
        self, vector, indices_vector, n_labels, G, h, A, b, I, B
    ):
        from cvxopt.glpk import ilp
        from numpy import array
        from cvxopt import matrix

        c = np.zeros((n_labels * (n_labels - 1) * 2, 1))
        for ind in range(len(vector)):
            c[ind, 0] = vector[ind]
        (_, x) = ilp(matrix(c), matrix(G), matrix(h), matrix(A), matrix(b), I, B)
        optX = array(x)
        # for indX in indexEmpty:
        #     optX[indX,0] = 0

        # #        epist_00 = 0
        # #        aleat_11 = 0
        #         scores = [0 for x in range(n_labels)]
        #         for i in range(n_labels):
        #             for k in range(0,i):
        #                 scores[i] += optX[indicesVector["%i_%i_%i"%(k,i,0)],0]
        #             for j in range(i+1,n_labels):
        #                 scores[i] += optX[indicesVector["%i_%i_%i"%(i,j,1)],0]
        # #                epist_00 += optX[indicesVector["%i_%i_%i"%(i,j,2)],0]
        # #                aleat_11 += optX[indicesVector["%i_%i_%i"%(i,j,3)],0]
        #         hard_prediction = [ind for ind in range(n_labels) if scores[ind] == 0]

        # Let both partial and preorder make the hard predictions in similar ways ...
        scores_d = [
            0 for x in range(n_labels)
        ]  # label i-th dominates at least one label
        scores_n = [0 for x in range(n_labels)]  # no label dominates label i-th
        for i in range(n_labels):
            for k in range(0, i):
                scores_d[i] += optX[indices_vector["%i_%i_%i" % (k, i, 1)], 0]
                scores_n[i] += optX[indices_vector["%i_%i_%i" % (k, i, 0)], 0]
            for j in range(i + 1, n_labels):
                scores_d[i] += optX[indices_vector["%i_%i_%i" % (i, j, 0)], 0]
                scores_n[i] += optX[indices_vector["%i_%i_%i" % (i, j, 1)], 0]
        #                epist_00 += optX[indicesVector["%i_%i_%i"%(i,j,2)],0]
        #                aleat_11 += optX[indicesVector["%i_%i_%i"%(i,j,3)],0]
        hard_prediction = [
            ind for ind in range(n_labels) if scores_d[ind] > 0 or scores_n[ind] == 0
        ]

        predicted_partialorder = []
        return hard_prediction, predicted_partialorder

    def _preorders(
        self,
        X_test,
        Y_test,
        known_indices,
        pairwise_4classifier,
        calibrated_2classifier,
    ):
        indices_vector = {}
        indVec = 0
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                for l in range(4):
                    key = "%i_%i_%i" % (i, j, l)
                    indices_vector[key] = indVec
                    indVec += 1
        G, h, A, b, I, B = predictors._preorders_compute_parameters(
            base_learner, indices_vector, n_labels
        )
        predicted_Y = []
        predicted_preorders = []
        n_instances, _ = X_test.shape
        for ind in range(n_instances):
            #            print(index, n_instances)
            vector = []
            #            indexEmpty = []
            for i in range(n_labels - 1):
                local_classifier = pairwise_4classifier[i]
                for k_2 in range(n_labels - i - 1):
                    # #                for j in range(i+1,n_labels):
                    #                     j = i + k_2 + 1
                    #                     if i in known_indices[ind] and j in known_indices[ind]:
                    #                         if Y_test[ind,i] + Y_test[ind,j] == 2:
                    #                             pairInfor = [0, 0, 0, 1]
                    #                         elif Y_test[ind,i] + Y_test[ind,j] == 0:
                    #                             pairInfor = [0, 0, 1, 0]
                    #                         elif Y_test[ind,i] > Y_test[ind,j]:
                    #                             pairInfor = [1, 0, 0, 0]
                    #                         elif Y_test[ind,i] < Y_test[ind,j]:
                    #                             pairInfor = [0, 1, 0,  0]
                    #                     else:
                    #                         clf = local_classifier[k_2]
                    #                         pairInfor_ori = clf.predict_proba(X_test[ind].reshape(1, -1))
                    #                         pairInfor = pairInfor_ori[0]
                    #                         presented_classes = list(clf.classes_)
                    #                         if len(presented_classes) < 4:
                    #                             pairInfor = [pairInfor[presented_classes.index(x)] if x in presented_classes else 0 for x in range(n_labels)]
                    #                         if i in known_indices[ind] and j not in known_indices[ind]:
                    #                             if Y_test[ind,i]  == 1:
                    #                                 pairInfor = [pairInfor[0], 0, 0, pairInfor[3]]
                    #                                 pairInfor = [x/sum(pairInfor) for x in pairInfor]
                    #                             elif Y_test[ind,i]  == 0:
                    #                                 pairInfor = [0, pairInfor[1], pairInfor[2], 0]
                    #                                 pairInfor = [x/sum(pairInfor) for x in pairInfor]
                    #                         elif i not in known_indices[ind] and j in known_indices[ind]:
                    #                             if Y_test[ind,j]  == 1:
                    #                                 pairInfor = [0, pairInfor[1], 0, pairInfor[3]]
                    #                                 pairInfor = [x/sum(pairInfor) for x in pairInfor]
                    #                             elif Y_test[ind,j]  == 0:
                    #                                 pairInfor = [pairInfor[0], 0, pairInfor[2], 0]
                    #                                 pairInfor = [x/sum(pairInfor) for x in pairInfor]
                    clf = local_classifier[k_2]
                    pairInfor_ori = clf.predict_proba(X_test[ind].reshape(1, -1))
                    pairInfor = pairInfor_ori[0]
                    presented_classes = list(clf.classes_)
                    if len(presented_classes) < 4:
                        pairInfor = [
                            (
                                pairInfor[presented_classes.index(x)]
                                if x in presented_classes
                                else 0
                            )
                            for x in range(n_labels)
                        ]

                    # add a small regularization term if the probabilistic prediction is deterministic instead of probabilistic
                    if max(pairInfor) == 1:
                        pairInfor = [
                            x - 10**-10 if x == 1 else (10**-10) / 3 for x in pairInfor
                        ]
                    if min(pairInfor) == 0:
                        zero_indices = [ind for ind in range(4) if pairInfor[ind] == 0]
                        pairInfor = [
                            (
                                (10**-10) / len(zero_indices)
                                if x == 0
                                else x - (10**-10) / (4 - len(zero_indices))
                            )
                            for x in pairInfor
                        ]
                    pairInfor = [-np.log(pairInfor[l]) for l in range(4)]
                    vector += pairInfor
            # Modify A to take into account the information of partial labes at the test time
            if len(known_indices[ind]) > 0:
                rowA = 0
                for i in range(n_labels - 1):
                    for j in range(i + 1, n_labels):
                        if i in known_indices[ind] and j in known_indices[ind]:
                            if Y_test[ind, i] + Y_test[ind, j] == 2:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 3)]] = 1
                            elif Y_test[ind, i] + Y_test[ind, j] == 0:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 3)]] = 0
                            elif Y_test[ind, i] > Y_test[ind, j]:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 3)]] = 0
                            elif Y_test[ind, i] < Y_test[ind, j]:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 3)]] = 0

                        elif i in known_indices[ind] and j not in known_indices[ind]:
                            if Y_test[ind, i] == 1:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 3)]] = 1
                            elif Y_test[ind, i] == 0:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 3)]] = 0

                        elif i not in known_indices[ind] and j in known_indices[ind]:
                            if Y_test[ind, j] == 1:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 3)]] = 1
                            elif Y_test[ind, j] == 0:
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 0)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 1)]] = 0
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 2)]] = 1
                                A[rowA, indices_vector["%i_%i_%i" % (i, j, 3)]] = 0

                        rowA += 1
            Gtest = np.array(G)
            Atest = np.array(A)
            # for indCol in indexEmpty:
            #     Gtest[:, indCol] = 0
            #     Atest[:, indCol] = 0
            hard_prediction_indices, predicted_preorder = (
                predictors._preorders_reasoning_procedure(
                    base_learner,
                    vector,
                    indices_vector,
                    n_labels,
                    Gtest,
                    h,
                    Atest,
                    b,
                    I,
                    B,
                )
            )
            hard_prediction = [
                1 if x in hard_prediction_indices else 0 for x in range(n_labels)
            ]
            predicted_Y.append(hard_prediction)
            predicted_preorders.append(predicted_preorder)
        return predicted_Y, predicted_preorders

    def _preorders_compute_parameters(self, indices_vector, n_labels):
        G = np.zeros(
            (n_labels * (n_labels - 1) * (n_labels - 2), n_labels * (n_labels - 1) * 2)
        )
        rowG = 0
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                for k in range(i):
                    indVecs = [
                        indices_vector[val]
                        for val in [
                            "%i_%i_%i" % (i, j, 0),
                            "%i_%i_%i" % (i, j, 3),
                            "%i_%i_%i" % (k, i, 1),
                            "%i_%i_%i" % (k, i, 3),
                            "%i_%i_%i" % (k, j, 0),
                            "%i_%i_%i" % (k, j, 3),
                        ]
                    ]
                    for ind in range(2):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(2, 6):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                    indVecs = [
                        indices_vector[val]
                        for val in [
                            "%i_%i_%i" % (i, j, 1),
                            "%i_%i_%i" % (i, j, 3),
                            "%i_%i_%i" % (k, i, 0),
                            "%i_%i_%i" % (k, i, 3),
                            "%i_%i_%i" % (k, j, 1),
                            "%i_%i_%i" % (k, j, 3),
                        ]
                    ]
                    for ind in range(2):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(2, 6):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                for k in range(i + 1, j):
                    indVecs = [
                        indices_vector[val]
                        for val in [
                            "%i_%i_%i" % (i, j, 0),
                            "%i_%i_%i" % (i, j, 3),
                            "%i_%i_%i" % (i, k, 0),
                            "%i_%i_%i" % (i, k, 3),
                            "%i_%i_%i" % (k, j, 0),
                            "%i_%i_%i" % (k, j, 3),
                        ]
                    ]
                    for ind in range(2):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(2, 6):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                    indVecs = [
                        indices_vector[val]
                        for val in [
                            "%i_%i_%i" % (i, j, 1),
                            "%i_%i_%i" % (i, j, 3),
                            "%i_%i_%i" % (i, k, 1),
                            "%i_%i_%i" % (i, k, 3),
                            "%i_%i_%i" % (k, j, 1),
                            "%i_%i_%i" % (k, j, 3),
                        ]
                    ]
                    for ind in range(2):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(2, 6):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                for k in range(j + 1, n_labels):
                    indVecs = [
                        indices_vector[val]
                        for val in [
                            "%i_%i_%i" % (i, j, 0),
                            "%i_%i_%i" % (i, j, 3),
                            "%i_%i_%i" % (i, k, 0),
                            "%i_%i_%i" % (i, k, 3),
                            "%i_%i_%i" % (j, k, 1),
                            "%i_%i_%i" % (j, k, 3),
                        ]
                    ]
                    for ind in range(2):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(2, 6):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                    indVecs = [
                        indices_vector[val]
                        for val in [
                            "%i_%i_%i" % (i, j, 1),
                            "%i_%i_%i" % (i, j, 3),
                            "%i_%i_%i" % (i, k, 1),
                            "%i_%i_%i" % (i, k, 3),
                            "%i_%i_%i" % (j, k, 0),
                            "%i_%i_%i" % (j, k, 3),
                        ]
                    ]
                    for ind in range(2):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(2, 6):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
        h = np.ones((n_labels * (n_labels - 1) * (n_labels - 2), 1))
        A = np.zeros(
            (int(n_labels * (n_labels - 1) * 0.5), int(n_labels * (n_labels - 1) * 2))
        )
        rowA = 0
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                # we can inject the information of partial labels at test time here
                for l in range(4):
                    indVec = indices_vector["%i_%i_%i" % (i, j, l)]
                    A[rowA, indVec] = 1
                rowA += 1
        b = np.ones((int(n_labels * (n_labels - 1) * 0.5), 1))
        I = set()
        B = set(range(n_labels * (n_labels - 1) * 2))
        return G, h, A, b, I, B

    def _preorders_reasoning_procedure(
        self, vector, indices_vector, n_labels, G, h, A, b, I, B
    ):
        from cvxopt.glpk import ilp
        from numpy import array
        from cvxopt import matrix

        c = np.zeros((n_labels * (n_labels - 1) * 2, 1))
        for ind in range(len(vector)):
            c[ind, 0] = vector[ind]
        (_, x) = ilp(matrix(c), matrix(G), matrix(h), matrix(A), matrix(b), I, B)
        optX = array(x)
        #        for indX in indexEmpty:
        #            optX[indX,0] = 0

        #        epist_00 = 0
        #        aleat_11 = 0
        scores_d = [
            0 for x in range(n_labels)
        ]  # label i-th dominates at least one label
        scores_n = [0 for x in range(n_labels)]  # no label dominates label i-th
        for i in range(n_labels):
            for k in range(0, i):
                scores_d[i] += optX[indices_vector["%i_%i_%i" % (k, i, 1)], 0]
                scores_n[i] += optX[indices_vector["%i_%i_%i" % (k, i, 0)], 0]
            for j in range(i + 1, n_labels):
                scores_d[i] += optX[indices_vector["%i_%i_%i" % (i, j, 0)], 0]
                scores_n[i] += optX[indices_vector["%i_%i_%i" % (i, j, 1)], 0]
        #                epist_00 += optX[indicesVector["%i_%i_%i"%(i,j,2)],0]
        #                aleat_11 += optX[indicesVector["%i_%i_%i"%(i,j,3)],0]
        hard_prediction = [
            ind for ind in range(n_labels) if scores_d[ind] > 0 or scores_n[ind] == 0
        ]
        predicted_preorder = []
        return hard_prediction, predicted_preorder

    def _hamming(self, predicted_Y, true_Y, known_indices):
        from sklearn.metrics import hamming_loss

        hamming = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            hard_pred = predicted_Y[index]
            true_lab = true_Y[index]
            hard_pred = [
                hard_pred[k] for k in range(n_labels) if k not in known_indices[index]
            ]
            true_lab = [
                true_lab[k] for k in range(n_labels) if k not in known_indices[index]
            ]
            hamming += 1 - hamming_loss(hard_pred, true_lab)
        return hamming / n_instances

    def _f1(self, predicted_Y, true_Y, known_indices):
        f1 = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            hard_pred = predicted_Y[index]
            true_lab = true_Y[index]
            hard_pred = [
                hard_pred[k] for k in range(n_labels) if k not in known_indices[index]
            ]
            true_lab = [
                true_lab[k] for k in range(n_labels) if k not in known_indices[index]
            ]
            if max(hard_pred) == 0 and max(true_lab) == 0:
                f1 += 1
            else:
                f1 += (2 * np.dot(hard_pred, true_lab)) / (
                    np.sum(hard_pred) + np.sum(true_lab)
                )
        return f1 / n_instances

    def _jaccard(self, predicted_Y, true_Y, known_indices):
        #        from sklearn.metrics import jaccard_score
        #        return np.mean(jaccard_score(hard_predictions, true_label, average=None))
        jaccard = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            hard_pred = predicted_Y[index]
            true_lab = true_Y[index]
            hard_pred = [
                hard_pred[k] for k in range(n_labels) if k not in known_indices[index]
            ]
            true_lab = [
                true_lab[k] for k in range(n_labels) if k not in known_indices[index]
            ]
            if max(hard_pred) == 0 and max(true_lab) == 0:
                jaccard += 1
            else:
                jaccard += (np.dot(hard_pred, true_lab)) / (
                    np.sum(hard_pred) + np.sum(true_lab) - np.dot(hard_pred, true_lab)
                )
        return jaccard / n_instances

    def _subset0_1(self, predicted_Y, true_Y, known_indices):
        #        from sklearn.metrics import jaccard_score
        #        return np.mean(jaccard_score(hard_predictions, true_label, average=None))
        subset0_1 = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            hard_pred = predicted_Y[index]
            true_lab = true_Y[index]
            hard_pred = [
                hard_pred[k] for k in range(n_labels) if k not in known_indices[index]
            ]
            true_lab = [
                true_lab[k] for k in range(n_labels) if k not in known_indices[index]
            ]
            if hard_pred == true_lab:
                subset0_1 += 1
        return subset0_1 / n_instances

    def _recall(self, predicted_Y, true_Y, known_indices):
        #        from sklearn.metrics import jaccard_score
        #        return np.mean(jaccard_score(hard_predictions, true_label, average=None))
        recall = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            hard_pred = predicted_Y[index]
            true_lab = true_Y[index]
            hard_pred = [
                hard_pred[k] for k in range(n_labels) if k not in known_indices[index]
            ]
            true_lab = [
                true_lab[k] for k in range(n_labels) if k not in known_indices[index]
            ]
            if np.dot(hard_pred, true_lab) == np.sum(true_lab):
                recall += 1
        return recall / n_instances


# for a quick test
if __name__ == "__main__":
    dataPath = "./data/"
    #    dataFile = 'emotions.arff'
    #    dataFile = 'scene.arff'
    #    dataFile = 'CHD_49.arff'
    # https://cometa.ujaen.es/datasets/VirusGO
    # https://www.uco.es/kdis/mllresources/#ImageDesc
    #    for dataFile in ['emotions.arff','CHD_49.arff', 'scene.arff']:
    #    n_labels_set = [14]
    label_ind = 0
    #    for dataFile in ['Water-quality.arff']:
    #    partial_prob = 0.1
    n_labels_set = [6, 6, 6, 14, 14]
    for dataFile in [
        "emotions.arff",
        "scene.arff",
        "CHD_49.arff",
        "Yeast.arff",
        "Water-quality.arff",
    ]:
        #    n_labels_set = [6, 6]
        #    for dataFile in ['emotions.arff', 'scene.arff']:
        n_labels = n_labels_set[label_ind]
        label_ind += 1
        #        n_labels = 6
        total_repeat = 1
        folds = 10
        for partial_prob in [0.1, 0.3]:
            for noisy_rate in [0.0, 0.2, 0.4]:
                for base_learner in ["RF", "ET"]:
                    #        for base_learner in ["RF", "ETC", "XGBoost", "LightGBM"]:
                    print(dataFile, base_learner)
                    #    base_learner = "LightGBM"
                    data = arff.loadarff(dataPath + dataFile)
                    df = pd.DataFrame(data[0]).to_numpy()
                    n_cols = len(df[0])
                    if dataFile in ["emotions.arff", "scene.arff"]:
                        X = df[:, : n_cols - n_labels]
                        Y = df[:, n_cols - n_labels :].astype(int)
                    else:
                        X = df[:, n_labels:]
                        Y = df[:, :n_labels].astype(int)
                    Y = np.where(Y < 0, 0, Y)

                    # from skmultilearn.dataset import load_from_arff

                    # features, labels = load_from_arff(dataPath+dataFile,
                    #     # number of labels
                    #     label_count=6,
                    #     # MULAN format, labels at the end of rows in arff data, using 'end' for label_location
                    #     # 'start' is also available for MEKA format
                    #     label_location='end',
                    #     # bag of words
                    #     input_feature_type='int', encode_nominal=False,
                    #     # sometimes the sparse ARFF loader is borked, like in delicious,
                    #     # scikit-multilearn converts the loaded data to sparse representations,
                    #     # so disabling the liac-arff sparse loader
                    #     # but you may set load_sparse to True if this fails
                    #     load_sparse=False,
                    #     # this decides whether to return attribute names or not, usually
                    #     # you don't need this
                    #     return_attribute_definitions=False)

                    for repeat in range(total_repeat):
                        average_hamming_loss_pairwise_2classifier = []
                        average_hamming_loss_pairwise_3classifier = []
                        average_hamming_loss_pairwise_4classifier = []
                        average_hamming_loss_ECC = []

                        average_f1_pairwise_2classifier = []
                        average_f1_pairwise_3classifier = []
                        average_f1_pairwise_4classifier = []
                        average_f1_ECC = []

                        average_jaccard_pairwise_2classifier = []
                        average_jaccard_pairwise_3classifier = []
                        average_jaccard_pairwise_4classifier = []
                        average_jaccard_ECC = []

                        average_subset0_1_pairwise_2classifier = []
                        average_subset0_1_pairwise_3classifier = []
                        average_subset0_1_pairwise_4classifier = []
                        average_subset0_1_ECC = []

                        average_recall_pairwise_2classifier = []
                        average_recall_pairwise_3classifier = []
                        average_recall_pairwise_4classifier = []
                        average_recall_ECC = []

                        fold = 0

                        Kf = KFold(n_splits=folds, random_state=42, shuffle=True)

                        for train_index, test_index in Kf.split(Y):
                            known_indices = []
                            for ind in range(len(test_index)):
                                known_index = []
                                for k in range(n_labels):
                                    if bernoulli.rvs(size=1, p=partial_prob)[0] == 1:
                                        if len(known_index) < n_labels - 1:
                                            known_index.append(k)
                                known_indices.append(known_index)

                            #                            print(aaaaaaaaaaaaa)
                            #                            known_indices = [[2] for index in range(len(test_index))] # for a quick test only

                            for index in range(len(train_index)):
                                for k in range(n_labels):
                                    if bernoulli.rvs(size=1, p=noisy_rate)[0] == 1:
                                        if Y[train_index[index], k] == 1:
                                            Y[train_index[index], k] = 0
                                        #                                          Y[train_index[index], k] = 1 # for additional test
                                        else:
                                            Y[train_index[index], k] = 1

                            print(["repeat", "fold", repeat, fold])
                            print(
                                "====================== pairwise_2classifier ======================"
                            )

                            pairwise_2classifier, calibrated_2classifier = (
                                BaseClassifiers._pairwise_2classifier(
                                    base_learner,
                                    n_labels,
                                    X[train_index],
                                    Y[train_index],
                                )
                            )
                            predicted_Y, predicted_ranks = predictors._CLR(
                                n_labels,
                                X[test_index],
                                pairwise_2classifier,
                                calibrated_2classifier,
                            )
                            hamming_loss_pairwise_2classifier = predictors._hamming(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_hamming_loss_pairwise_2classifier.append(
                                hamming_loss_pairwise_2classifier
                            )
                            print(hamming_loss_pairwise_2classifier)
                            f1_pairwise_2classifier = predictors._f1(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_f1_pairwise_2classifier.append(
                                f1_pairwise_2classifier
                            )
                            print(f1_pairwise_2classifier)
                            jaccard_pairwise_2classifier = predictors._jaccard(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_jaccard_pairwise_2classifier.append(
                                jaccard_pairwise_2classifier
                            )
                            print(jaccard_pairwise_2classifier)
                            subset0_1_pairwise_2classifier = predictors._subset0_1(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_subset0_1_pairwise_2classifier.append(
                                subset0_1_pairwise_2classifier
                            )
                            print(subset0_1_pairwise_2classifier)
                            recall_pairwise_2classifier = predictors._recall(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_recall_pairwise_2classifier.append(
                                recall_pairwise_2classifier
                            )
                            print(recall_pairwise_2classifier)
                            #                    print(np.mean([np.mean(x) for x in hard_predictions]))

                            print(
                                "====================== pairwise_3classifier ======================"
                            )
                            pairwise_3classifier = (
                                BaseClassifiers._pairwise_3classifier(
                                    base_learner,
                                    n_labels,
                                    X[train_index],
                                    Y[train_index],
                                )
                            )
                            predicted_Y, predicted_ranks = predictors._partialorders(
                                n_labels,
                                X[test_index],
                                Y[test_index],
                                known_indices,
                                pairwise_3classifier,
                                calibrated_2classifier,
                            )
                            hamming_loss_pairwise_3classifier = predictors._hamming(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_hamming_loss_pairwise_3classifier.append(
                                hamming_loss_pairwise_3classifier
                            )
                            print(hamming_loss_pairwise_3classifier)
                            f1_pairwise_3classifier = predictors._f1(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_f1_pairwise_3classifier.append(
                                f1_pairwise_3classifier
                            )
                            print(f1_pairwise_3classifier)
                            jaccard_pairwise_3classifier = predictors._jaccard(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_jaccard_pairwise_3classifier.append(
                                jaccard_pairwise_3classifier
                            )
                            print(jaccard_pairwise_3classifier)
                            subset0_1_pairwise_3classifier = predictors._subset0_1(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_subset0_1_pairwise_3classifier.append(
                                subset0_1_pairwise_3classifier
                            )
                            print(subset0_1_pairwise_3classifier)
                            recall_pairwise_3classifier = predictors._recall(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_recall_pairwise_3classifier.append(
                                recall_pairwise_3classifier
                            )
                            print(recall_pairwise_3classifier)

                            #                    print(np.mean([np.mean(x) for x in hard_predictions]))

                            print(
                                "====================== pairwise_4classifier ======================"
                            )
                            pairwise_4classifier = (
                                BaseClassifiers._pairwise_4classifier(
                                    base_learner,
                                    n_labels,
                                    X[train_index],
                                    Y[train_index],
                                )
                            )
                            predicted_Y, predicted_ranks = predictors._preorders(
                                n_labels,
                                X[test_index],
                                Y[test_index],
                                known_indices,
                                pairwise_4classifier,
                                calibrated_2classifier,
                            )
                            hamming_loss_pairwise_4classifier = predictors._hamming(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_hamming_loss_pairwise_4classifier.append(
                                hamming_loss_pairwise_4classifier
                            )
                            print(hamming_loss_pairwise_4classifier)
                            f1_pairwise_4classifier = predictors._f1(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_f1_pairwise_4classifier.append(
                                f1_pairwise_4classifier
                            )
                            print(f1_pairwise_4classifier)
                            jaccard_pairwise_4classifier = predictors._jaccard(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_jaccard_pairwise_4classifier.append(
                                jaccard_pairwise_4classifier
                            )
                            print(jaccard_pairwise_4classifier)
                            subset0_1_pairwise_4classifier = predictors._subset0_1(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_subset0_1_pairwise_4classifier.append(
                                subset0_1_pairwise_4classifier
                            )
                            print(subset0_1_pairwise_4classifier)
                            recall_pairwise_4classifier = predictors._recall(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_recall_pairwise_4classifier.append(
                                recall_pairwise_4classifier
                            )
                            print(recall_pairwise_4classifier)

                            print("====================== ECC ======================")
                            from sklearn.multioutput import ClassifierChain
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.ensemble import ExtraTreesClassifier
                            from sklearn.ensemble import GradientBoostingClassifier
                            import lightgbm as lgb

                            if base_learner == "RF":
                                clf = RandomForestClassifier(random_state=42)
                            if base_learner == "ET":
                                clf = ExtraTreesClassifier(random_state=42)
                            if base_learner == "XGBoost":
                                clf = GradientBoostingClassifier(random_state=42)
                            if base_learner == "LightGBM":
                                clf = lgb.LGBMClassifier(random_state=42)
                            chains = [
                                ClassifierChain(clf, order="random", random_state=i)
                                for i in range(10)
                            ]
                            for chain in chains:
                                chain.fit(X[train_index], Y[train_index])
                            Y_pred_chains = np.array(
                                [chain.predict(X[test_index]) for chain in chains]
                            )
                            Y_pred_ensemble = Y_pred_chains.mean(axis=0)
                            predicted_Y = np.where(Y_pred_ensemble > 0.5, 1, 0)

                            #                    ECC.fit(X[train_index].astype(float), Y[train_index].astype(float))
                            #                    predicted_Y = ECC.predict(X[test_index].astype(float))
                            print("====================== ECC ======================")
                            hamming_loss_ECC = predictors._hamming(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_hamming_loss_ECC.append(hamming_loss_ECC)
                            print(hamming_loss_ECC)
                            f1_ECC = predictors._f1(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_f1_ECC.append(f1_ECC)
                            print(f1_ECC)
                            jaccard_ECC = predictors._jaccard(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_jaccard_ECC.append(jaccard_ECC)
                            print(jaccard_ECC)
                            subset0_1_ECC = predictors._subset0_1(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_subset0_1_ECC.append(subset0_1_ECC)
                            print(subset0_1_ECC)
                            recall_ECC = predictors._recall(
                                base_learner, predicted_Y, Y[test_index], known_indices
                            )
                            average_recall_ECC.append(recall_ECC)
                            print(recall_ECC)

                            fold += 1

                        print(
                            "====================== Average results: ======================"
                        )
                        print("====================== Haming ======================")
                        print(np.mean(average_hamming_loss_pairwise_2classifier))
                        print(np.mean(average_hamming_loss_pairwise_3classifier))
                        print(np.mean(average_hamming_loss_pairwise_4classifier))
                        print(np.mean(average_hamming_loss_ECC))
                        print("====================== F1 ======================")
                        print(np.mean(average_f1_pairwise_2classifier))
                        print(np.mean(average_f1_pairwise_3classifier))
                        print(np.mean(average_f1_pairwise_4classifier))
                        print(np.mean(average_f1_ECC))
                        print("====================== Jaccard ======================")
                        print(np.mean(average_jaccard_pairwise_2classifier))
                        print(np.mean(average_jaccard_pairwise_3classifier))
                        print(np.mean(average_jaccard_pairwise_4classifier))
                        print(np.mean(average_jaccard_ECC))
                        print(
                            "====================== Subset 0/1 ======================"
                        )
                        print(np.mean(average_subset0_1_pairwise_2classifier))
                        print(np.mean(average_subset0_1_pairwise_3classifier))
                        print(np.mean(average_subset0_1_pairwise_4classifier))
                        print(np.mean(average_subset0_1_ECC))
                        print("====================== Recall ======================")
                        print(np.mean(average_recall_pairwise_2classifier))
                        print(np.mean(average_recall_pairwise_3classifier))
                        print(np.mean(average_recall_pairwise_4classifier))
                        print(np.mean(average_recall_ECC))

                    final_results = [
                        [
                            np.mean(average_hamming_loss_pairwise_2classifier),
                            np.std(average_hamming_loss_pairwise_2classifier),
                            np.mean(average_hamming_loss_pairwise_3classifier),
                            np.std(average_hamming_loss_pairwise_3classifier),
                            np.mean(average_hamming_loss_pairwise_4classifier),
                            np.std(average_hamming_loss_pairwise_4classifier),
                            np.mean(average_hamming_loss_ECC),
                            np.std(average_hamming_loss_ECC),
                        ],
                        [
                            np.mean(average_f1_pairwise_2classifier),
                            np.std(average_f1_pairwise_2classifier),
                            np.mean(average_f1_pairwise_3classifier),
                            np.std(average_f1_pairwise_3classifier),
                            np.mean(average_f1_pairwise_4classifier),
                            np.std(average_f1_pairwise_4classifier),
                            np.mean(average_f1_ECC),
                            np.std(average_f1_ECC),
                        ],
                        [
                            np.mean(average_jaccard_pairwise_2classifier),
                            np.std(average_jaccard_pairwise_2classifier),
                            np.mean(average_jaccard_pairwise_3classifier),
                            np.std(average_jaccard_pairwise_3classifier),
                            np.mean(average_jaccard_pairwise_4classifier),
                            np.std(average_jaccard_pairwise_4classifier),
                            np.mean(average_jaccard_ECC),
                            np.std(average_jaccard_ECC),
                        ],
                        [
                            np.mean(average_subset0_1_pairwise_2classifier),
                            np.std(average_subset0_1_pairwise_2classifier),
                            np.mean(average_subset0_1_pairwise_3classifier),
                            np.std(average_subset0_1_pairwise_3classifier),
                            np.mean(average_subset0_1_pairwise_4classifier),
                            np.std(average_subset0_1_pairwise_4classifier),
                            np.mean(average_subset0_1_ECC),
                            np.std(average_subset0_1_ECC),
                        ],
                        [
                            np.mean(average_recall_pairwise_2classifier),
                            np.std(average_recall_pairwise_2classifier),
                            np.mean(average_recall_pairwise_3classifier),
                            np.std(average_recall_pairwise_3classifier),
                            np.mean(average_recall_pairwise_4classifier),
                            np.std(average_recall_pairwise_4classifier),
                            np.mean(average_recall_ECC),
                            np.std(average_recall_ECC),
                        ],
                    ]

                    res_file = (
                        "addition_partial_noisy_4_w_try_compareAcc_%i_%i_%i_%i_%s_%s"
                        % (
                            int(noisy_rate * 10),
                            int(partial_prob * 10),
                            total_repeat,
                            folds,
                            dataFile,
                            base_learner,
                        )
                    )
                    file = open(res_file, "w")
                    file.writelines("%s\n" % line for line in final_results)
                    file.close()
