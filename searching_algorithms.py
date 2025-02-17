from cvxopt.glpk import ilp
import numpy as np
from numpy import array
from cvxopt import matrix

from constants import TargetMetric


class Search_BOPreOs:
    """
    Search for the best preference order using the BOPreOs model.
    """

    # For ilp with cvxopt.glpk, see https://gist.github.com/nipunbatra/7059160?fbclid=IwY2xjawEgKndleHRuA2FlbQIxMAABHfelujAwksDZjWb9Pn-Nfakv52P-ltg295HO8m0F_XnTjbp9_TvAlZxcjA_aem_ZA3BrIWlMvWu2HlK8bKKaA

    def __init__(
        self,
        pairwise_probabilistic_predictions,
        n_labels,
        n_instances,
        target_metric: TargetMetric,
        height,
    ):
        self.pairwise_probabilistic_predictions = pairwise_probabilistic_predictions
        self.n_labels = n_labels
        self.n_instances = n_instances
        self.height = height
        self.target_metric = target_metric

    # PRE-ORDER
    # subset - height = 2
    # subset - height = None
    # hamming - height = 2
    # hamming - height = None

    # PARTIAL-ORDER
    # subset - height = 2
    # subset - height = None
    # hamming - height = 2
    # hamming - height = None

    def PRE_ORDER(self):
        print(f"Search_BOPreOs: {self.target_metric}")

        indices_vector = {}
        indVec = 0
        for i in range(self.n_labels - 1):
            for j in range(i + 1, self.n_labels):
                for l in range(4):
                    #                    key = "%i_%i_%i" % (i, j, l)
                    indices_vector[f"{i}_{j}_{l}"] = indVec
                    indVec += 1
        G, h, A, b, I, B = self._encode_parameters_PRE_ORDER(indices_vector)  # type: ignore
        predicted_Y = []
        predicted_preorders = []
        for n in range(self.n_instances):
            #            print(index, n_instances)
            vector = []
            #           indexEmpty = []
            if self.target_metric == TargetMetric.Hamming:
                for i in range(self.n_labels - 1):
                    for j in range(i + 1, self.n_labels):
                        pairInfor = [
                            -self.pairwise_probabilistic_predictions[f"{i}_{j}_{n}_{l}"]
                            for l in range(4)
                        ]
                        vector += pairInfor
            elif self.target_metric == TargetMetric.Subset:
                for i in range(self.n_labels - 1):
                    for j in range(i + 1, self.n_labels):
                        pairInfor = [
                            -np.log(
                                self.pairwise_probabilistic_predictions[
                                    f"{i}_{j}_{n}_{l}"
                                ]
                            )
                            for l in range(4)
                        ]
                        vector += pairInfor
            else:
                raise ValueError(f"Unknown target metric: {self.target_metric}")
            Gtest = np.array(G)
            Atest = np.array(A)
            #            for indCol in indexEmpty:
            #                Gtest[:, indCol] = 0
            #                Atest[:, indCol] = 0
            hard_prediction_indices, predicted_preorder = (
                self._reasoning_procedure_PRE_ORDER(
                    vector,
                    indices_vector,
                    self.n_labels,
                    Gtest,
                    h,
                    Atest,
                    b,
                    I,
                    B,
                )
            )
            # , indexEmpty)
            hard_prediction = [
                1 if x in hard_prediction_indices else 0 for x in range(self.n_labels)
            ]
            predicted_Y.append(hard_prediction)
            predicted_preorders.append(predicted_preorder)
        return predicted_Y, predicted_preorders, indices_vector

    def _encode_parameters_PRE_ORDER(self, indices_vector):
        assert self.n_labels is not None
        h = np.ones((self.n_labels * (self.n_labels - 1) * (self.n_labels - 2), 1))
        A = np.zeros(
            (
                int(self.n_labels * (self.n_labels - 1) * 0.5),
                int(self.n_labels * (self.n_labels - 1) * 2),
            )
        )
        rowA = 0
        for i in range(self.n_labels - 1):
            for j in range(i + 1, self.n_labels):
                # we can inject the information of partial labels at test time here
                for l in range(4):
                    indVec = indices_vector["%i_%i_%i" % (i, j, l)]
                    A[rowA, indVec] = 1
                rowA += 1
        b = np.ones((int(self.n_labels * (self.n_labels - 1) * 0.5), 1))
        I = set()
        B = set(range(self.n_labels * (self.n_labels - 1) * 2))
        G = np.zeros(
            (
                self.n_labels * (self.n_labels - 1) * (self.n_labels - 2),
                self.n_labels * (self.n_labels - 1) * 2,
            )
        )

        if not self.height:
            rowG = 0
            for i in range(self.n_labels - 1):
                for j in range(i + 1, self.n_labels):
                    for k in range(i):
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{i}_{j}_{0}",
                                f"{i}_{j}_{3}",
                                f"{k}_{i}_{1}",
                                f"{k}_{i}_{3}",
                                f"{k}_{j}_{0}",
                                f"{k}_{j}_{3}",
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
                                f"{i}_{j}_{1}",
                                f"{i}_{j}_{3}",
                                f"{k}_{i}_{0}",
                                f"{k}_{i}_{3}",
                                f"{k}_{j}_{1}",
                                f"{k}_{j}_{3}",
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
                                f"{i}_{j}_{0}",
                                f"{i}_{j}_{3}",
                                f"{i}_{k}_{0}",
                                f"{i}_{k}_{3}",
                                f"{k}_{j}_{0}",
                                f"{k}_{j}_{3}",
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
                                f"{i}_{j}_{1}",
                                f"{i}_{j}_{3}",
                                f"{i}_{k}_{1}",
                                f"{i}_{k}_{3}",
                                f"{k}_{j}_{1}",
                                f"{k}_{j}_{3}",
                            ]
                        ]
                        for ind in range(2):
                            G[rowG, indVecs[ind]] = -1
                        for ind in range(2, 6):
                            G[rowG, indVecs[ind]] = 1
                        rowG += 1
                    for k in range(j + 1, self.n_labels):
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{i}_{j}_{0}",
                                f"{i}_{j}_{3}",
                                f"{i}_{k}_{0}",
                                f"{i}_{k}_{3}",
                                f"{j}_{k}_{1}",
                                f"{j}_{k}_{3}",
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
                                f"{i}_{j}_{1}",
                                f"{i}_{j}_{3}",
                                f"{i}_{k}_{1}",
                                f"{i}_{k}_{3}",
                                f"{j}_{k}_{0}",
                                f"{j}_{k}_{3}",
                            ]
                        ]
                        for ind in range(2):
                            G[rowG, indVecs[ind]] = -1
                        for ind in range(2, 6):
                            G[rowG, indVecs[ind]] = 1
                        rowG += 1
            return G, h, A, b, I, B

        elif self.height == 2:
            rowG = 0
            for i in range(self.n_labels - 1):
                for j in range(i + 1, self.n_labels):
                    for k in range(i):
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{k}_{i}_{1}",
                                f"{k}_{i}_{3}",
                                f"{k}_{j}_{0}",
                                f"{k}_{j}_{3}",
                            ]
                        ]
                        for ind_tupe in indVecs:
                            G[rowG, ind_tupe] = 1
                        rowG += 1

                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{k}_{i}_{0}",
                                f"{k}_{i}_{3}",
                                f"{k}_{j}_{1}",
                                f"{k}_{j}_{3}",
                            ]
                        ]
                        for ind_tupe in indVecs:
                            G[rowG, ind_tupe] = 1
                        rowG += 1
                    for k in range(i + 1, j):
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{i}_{k}_{0}",
                                f"{i}_{k}_{3}",
                                f"{k}_{j}_{0}",
                                f"{k}_{j}_{3}",
                            ]
                        ]
                        for ind_tupe in indVecs:
                            G[rowG, ind_tupe] = 1
                        rowG += 1
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{i}_{k}_{1}",
                                f"{i}_{k}_{3}",
                                f"{k}_{j}_{1}",
                                f"{k}_{j}_{3}",
                            ]
                        ]
                        for ind_tupe in indVecs:
                            G[rowG, ind_tupe] = 1
                        rowG += 1
                    for k in range(j + 1, self.n_labels):
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{i}_{k}_{0}",
                                f"{i}_{k}_{3}",
                                f"{j}_{k}_{1}",
                                f"{j}_{k}_{3}",
                            ]
                        ]
                        for ind_tupe in indVecs:
                            G[rowG, ind_tupe] = 1
                        rowG += 1
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{i}_{k}_{1}",
                                f"{i}_{k}_{3}",
                                f"{j}_{k}_{0}",
                                f"{j}_{k}_{3}",
                            ]
                        ]
                        for ind_tupe in indVecs:
                            G[rowG, ind_tupe] = 1
                        rowG += 1
            return G, h, A, b, I, B
        else:
            raise ValueError("The height is not supported")

    def _reasoning_procedure_PRE_ORDER(
        self, vector, indices_vector, n_labels, G, h, A, b, I, B
    ):
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
                scores_d[i] += optX[indices_vector[f"{k}_{i}_{1}"], 0]
                scores_n[i] += optX[indices_vector[f"{k}_{i}_{0}"], 0]
            for j in range(i + 1, n_labels):
                scores_d[i] += optX[indices_vector[f"{i}_{j}_{0}"], 0]
                scores_n[i] += optX[indices_vector[f"{i}_{j}_{1}"], 0]
        #                epist_00 += optX[indicesVector["%i_%i_%i"%(i,j,2)],0]
        #                aleat_11 += optX[indicesVector["%i_%i_%i"%(i,j,3)],0]
        hard_prediction = [
            ind for ind in range(n_labels) if scores_d[ind] > 0 or scores_n[ind] == 0
        ]
        # optX: [n*(n-1)*2, 1] -> [1]
        #  [[0.1], [0.2],] => [0.1, 0.2]
        predicted_partial_order = optX.flatten()
        return hard_prediction, predicted_partial_order


class Search_BOParOs:
    def __init__(
        self,
        pairwise_probabilistic_predictions,
        n_labels,
        n_instances,
        target_metric,
        height,
    ):
        self.pairwise_probabilistic_predictions = pairwise_probabilistic_predictions
        self.n_labels = n_labels
        self.n_instances = n_instances
        self.height = height
        self.target_metric = target_metric

    def PARTIAL_ORDER(self):
        indices_vector = {}
        indVec = 0
        # How to make sure self.n_labels is not None
        assert self.n_labels is not None

        for i in range(self.n_labels - 1):
            for j in range(i + 1, self.n_labels):
                for l in range(3):
                    #                    key = "%i_%i_%i" % (i, j, l)
                    indices_vector[f"{i}_{j}_{l}"] = indVec
                    indVec += 1
        G, h, A, b, I, B = self._encode_parameters_PARTIAL_ORDER(indices_vector)  # type: ignore
        predicted_Y = []
        predicted_partial_orders = []
        for n in range(self.n_instances):
            vector = []
            if self.target_metric == TargetMetric.Hamming:
                for i in range(self.n_labels - 1):
                    for j in range(i + 1, self.n_labels):
                        pairInfor = [
                            -self.pairwise_probabilistic_predictions[f"{i}_{j}_{n}_{l}"]
                            for l in range(3)
                        ]
                        vector += pairInfor
            elif self.target_metric == TargetMetric.Subset:
                for i in range(self.n_labels - 1):
                    for j in range(i + 1, self.n_labels):
                        pairInfor = [
                            -np.log(
                                self.pairwise_probabilistic_predictions[
                                    f"{i}_{j}_{n}_{l}"
                                ]
                            )
                            for l in range(3)
                        ]
                        vector += pairInfor
            else:
                raise ValueError(f"Unknown target metric: {self.target_metric}")
            Gtest = np.array(G)
            Atest = np.array(A)
            hard_prediction_indices, predicted_partial_order = (
                self._reasoning_procedure_PARTIAL_ORDER(
                    vector,
                    indices_vector,
                    Gtest,
                    h,
                    Atest,
                    b,
                    I,
                    B,
                )
            )
            hard_prediction = [
                1 if x in hard_prediction_indices else 0 for x in range(self.n_labels)
            ]
            predicted_Y.append(hard_prediction)
            predicted_partial_orders.append(predicted_partial_order)
        return predicted_Y, predicted_partial_orders, indices_vector

    def _encode_parameters_PARTIAL_ORDER(self, indices_vector):
        assert self.n_labels is not None

        h = np.ones((self.n_labels * (self.n_labels - 1) * (self.n_labels - 2), 1))
        A = np.zeros(
            (
                int(self.n_labels * (self.n_labels - 1) * 0.5),
                int(self.n_labels * (self.n_labels - 1) * 1.5),
            )
        )
        rowA = 0
        for i in range(self.n_labels - 1):
            for j in range(i + 1, self.n_labels):
                # we can inject the information of partial labels at test time here
                for l in range(3):
                    indVec = indices_vector[f"{i}_{j}_{l}"]
                    A[rowA, indVec] = 1
                rowA += 1
        b = np.ones((int(self.n_labels * (self.n_labels - 1) * 0.5), 1))
        I = set()
        B = set(range(int(self.n_labels * (self.n_labels - 1) * 1.5)))
        G = np.zeros(
            (
                int(self.n_labels * (self.n_labels - 1) * (self.n_labels - 2)),
                int(self.n_labels * (self.n_labels - 1) * 1.5),
            )
        )

        if not self.height:
            rowG = 0
            for i in range(self.n_labels - 1):
                for j in range(i + 1, self.n_labels):
                    for k in range(i):
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{i}_{j}_{0}",
                                f"{k}_{i}_{1}",
                                f"{k}_{j}_{0}",
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
                                f"{i}_{j}_{1}",
                                f"{k}_{i}_{0}",
                                f"{k}_{j}_{1}",
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
                                f"{i}_{j}_{0}",
                                f"{i}_{k}_{0}",
                                f"{k}_{j}_{0}",
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
                                f"{i}_{j}_{1}",
                                f"{i}_{k}_{1}",
                                f"{k}_{j}_{1}",
                            ]
                        ]
                        for ind in range(1):
                            G[rowG, indVecs[ind]] = -1
                        for ind in range(1, 3):
                            G[rowG, indVecs[ind]] = 1
                        rowG += 1
                    for k in range(j + 1, self.n_labels):
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{i}_{j}_{0}",
                                f"{i}_{k}_{0}",
                                f"{j}_{k}_{1}",
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
                                f"{i}_{j}_{1}",
                                f"{i}_{k}_{1}",
                                f"{j}_{k}_{0}",
                            ]
                        ]
                        for ind in range(1):
                            G[rowG, indVecs[ind]] = -1
                        for ind in range(1, 3):
                            G[rowG, indVecs[ind]] = 1
                        rowG += 1

            return G, h, A, b, I, B

        elif self.height == 2:
            rowG = 0
            for i in range(self.n_labels - 1):
                for j in range(i + 1, self.n_labels):
                    for k in range(i):
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{k}_{i}_{1}",
                                f"{k}_{j}_{0}",
                            ]
                        ]
                        for ind_tupe in indVecs:
                            G[rowG, ind_tupe] = 1
                        rowG += 1
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{k}_{i}_{0}",
                                f"{k}_{j}_{1}",
                            ]
                        ]
                        for ind_tupe in indVecs:
                            G[rowG, ind_tupe] = 1
                        rowG += 1
                    for k in range(i + 1, j):
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{i}_{k}_{0}",
                                f"{k}_{j}_{0}",
                            ]
                        ]
                        for ind_tupe in indVecs:
                            G[rowG, ind_tupe] = 1
                        rowG += 1
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{i}_{k}_{1}",
                                f"{k}_{j}_{1}",
                            ]
                        ]
                        for ind_tupe in indVecs:
                            G[rowG, ind_tupe] = 1
                        rowG += 1
                    for k in range(j + 1, self.n_labels):
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{i}_{k}_{0}",
                                f"{j}_{k}_{1}",
                            ]
                        ]
                        for ind_tupe in indVecs:
                            G[rowG, ind_tupe] = 1
                        rowG += 1
                        indVecs = [
                            indices_vector[val]
                            for val in [
                                f"{i}_{k}_{1}",
                                f"{j}_{k}_{0}",
                            ]
                        ]
                        for ind_tupe in indVecs:
                            G[rowG, ind_tupe] = 1
                        rowG += 1

            return G, h, A, b, I, B
        else:
            raise ValueError("The height is not supported")

    def _reasoning_procedure_PARTIAL_ORDER(
        self, vector, indices_vector, G, h, A, b, I, B
    ):
        c = np.zeros((self.n_labels * (self.n_labels - 1) * 2, 1))
        for ind in range(len(vector)):
            c[ind, 0] = vector[ind]
        (_, x) = ilp(matrix(c), matrix(G), matrix(h), matrix(A), matrix(b), I, B)
        optX = array(x)
        #        for indX in indexEmpty:
        #            optX[indX,0] = 0

        # Let both partial and preorder make the hard predictions in similar ways ...
        scores_d = [
            0 for x in range(self.n_labels)
        ]  # label i-th dominates at least one label
        scores_n = [0 for x in range(self.n_labels)]  # no label dominates label i-th
        for i in range(self.n_labels):
            for k in range(0, i):
                scores_d[i] += optX[indices_vector[f"{k}_{i}_{1}"], 0]
                scores_n[i] += optX[indices_vector[f"{k}_{i}_{0}"], 0]
            for j in range(i + 1, self.n_labels):
                scores_d[i] += optX[indices_vector[f"{i}_{j}_{0}"], 0]
                scores_n[i] += optX[indices_vector[f"{i}_{j}_{1}"], 0]
        #                epist_00 += optX[indicesVector["%i_%i_%i"%(i,j,2)],0]
        #                aleat_11 += optX[indicesVector["%i_%i_%i"%(i,j,3)],0]
        hard_prediction = [
            ind
            for ind in range(self.n_labels)
            if scores_d[ind] > 0 or scores_n[ind] == 0
        ]

        # optX: [n*(n-1)*2, 1] -> [1]
        #  [[0.1], [0.2],] => [0.1, 0.2]
        predicted_partial_order = optX.flatten()
        return hard_prediction, predicted_partial_order
