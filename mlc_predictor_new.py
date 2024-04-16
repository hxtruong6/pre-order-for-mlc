from inference_metric import PreferenceOrder
from base_classifer import BaseClassifiers


class MLCPredictor:
    def __init__(self, classifer: BaseClassifiers, preference_order: PreferenceOrder):
        self.classifer = classifer
        self.preference_order = preference_order

    def predict(self, X, **kwargs):
        pass


class PredictPartialOrder(MLCPredictor):
    def __init__(self, classifer: BaseClassifiers, *args):
        super().__init__(classifer, preference_order=PreferenceOrder.PARTIAL_ORDER)
        self.pairwise_classifiers = None
        self.n_labels = None
        self.n_instances = None

    def fit(self, X, Y):
        self.n_labels = len(Y[0])
        self.n_instances, _ = Y.shape
        self.pairwise_classifiers = self.classifer.pairwise_partial_order_classifer(
            X, Y
        )

    def predict(self, X_test, metric="subset", **kwargs):
        print(f"Inference partial order for metric: {metric}")

        if self.pairwise_classifiers is None:
            raise ValueError(
                "pairwise_classifiers is None. Please run fit(X, Y) before predict(X, Y, X_test, metric)"
            )

        pairwise_probabilsitic_predictions = {}

        assert self.n_labels is not None
        assert self.n_instances is not None

        #  learn probabilistic predictions for each pair of labels
        for i in range(self.n_labels - 1):
            for j in range(i + 1, self.n_labels):
                key_classifier = f"{i}_{j}"
                probabilistic_prediction = self.pairwise_classifiers[
                    key_classifier
                ].predict_proba(
                    X_test
                )  # --> Nx3 numpy.array
                # It can happen that the training set of self.pairwise_classifiers[key_classifier]
                # contains less than 3 classes. In this case, we should ensure that the unseen classes
                # should have predicted probability equal 0.
                # list(self.pairwise_classifiers[key_classifier].classes_) gives us the list of classes
                # which appear on at least one training instance, which can contain either 1 or 2 or 3 classes

                for n in range(self.n_instances):

                    for l in range(3):
                        pairwise_probabilsitic_predictions[f"{i}_{j}_{n}_{l}"] = (
                            probabilistic_prediction[n, l]
                        )

            #   pairwise_probabilsitic_predictions = {}
            #
            #
            #
            #            call the pairwise classifier pairwise[key_classifier] for the label pair (y_i,y_j)
            #            which has been store when runing inference_metric.fit(X_train, Y_train)
            #            for n in range(n_test_instance):
            #                if partial_order:
            #                   for l in range(3)
            #                       key_pairwise_probabilsitic_predictions = "%i_%i_%i_%i" % (i, j, n,l)
            #                       pairwise_probabilsitic_predictions[key_pairwise_probabilsitic_predictions] = the l prediction
            #                       of classifier[key_classifier] on the n-th test instance (classifier[key_classifier] is a 3-class classifer)
            #                if pre_order:
            #                   for l in range(4)
            #                       key_pairwise_probabilsitic_predictions = "%i_%i_%i_%i" % (i, j, n,l)
            #                       pairwise_probabilsitic_predictions[key_pairwise_probabilsitic_predictions] = the l prediction
            #                       of classifier[key_classifier] on the n-th test instance (classifier[key_classifier] is a 4-class classifer)

        if metric == "subset":
            # after having the probabilistic predictions,
            # denoted by pairwise_probabilsitic_predictions,
            # we can use them to infer the partial order w.r.t either subset 0/1 accuracy or Hamming accuracy
            return self.inference_subset(
                pairwise_probabilsitic_predictions, height=None
            )
        elif metric == "hamming":
            pass
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def inference_subset(
        self, pairwise_probabilsitic_predictions, height=None
    ):  # = _predict_PaOr_Subset
        # if height == None -> unrestricted structure
        indices_vector = {}
        indVec = 0
        # How to make sure self.n_labels is not None
        assert self.n_labels is not None

        for i in range(self.n_labels - 1):
            for j in range(i + 1, self.n_labels):
                for l in range(3):
                    key = "%i_%i_%i" % (i, j, l)
                    indices_vector[key] = indVec
                    indVec += 1
        G, h, A, b, I, B = self._encode_parameters_PaOr_Subset(indices_vector, height)
        predicted_Y = []
        predicted_preorders = []
        n_instances, _ = X_test.shape
        for index in range(n_instances):
            #            print(index, n_instances)
            vector = []
            #            indexEmpty = []
            for i in range(n_labels - 1):
                local_classifier = pairwise_3classifier[i]
                for k_2 in range(n_labels - i - 1):
                    clf = local_classifier[k_2]
                    j = i + k_2 + 1
                    pairInfor_ori = clf.predict_proba(X_test[index].reshape(1, -1))
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
            #                    empty = [indices_vector["%i_%i_%i"%(i,j,l)] for l in range(3) if pairInfor[l] == 0]
            #                    indexEmpty += empty
            Gtest = np.array(G)
            Atest = np.array(A)
            #            for indCol in indexEmpty:
            #                Gtest[:, indCol] = 0
            #                Atest[:, indCol] = 0
            hard_prediction_indices, predicted_preorder = (
                MLCPredictor._partialorders_reasoning_procedure(
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
            # , indexEmpty)
            hard_prediction = [
                1 if x in hard_prediction_indices else 0 for x in range(n_labels)
            ]
            predicted_Y.append(hard_prediction)
            predicted_preorders.append(predicted_preorder)
        return predicted_Y, predicted_preorders

    def _encode_parameters_PaOr_Subset(self, indices_vector, height=None):
        n_labels = self.n_labels
        assert n_labels is not None

        if not height:
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
                (
                    int(n_labels * (n_labels - 1) * 0.5),
                    int(n_labels * (n_labels - 1) * 1.5),
                )
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

        elif height == 2:

            pass
            # return G, h, A, b, I, B
        else:
            raise ValueError("The height is not supported")

    def _reasoning_procedure_PaOr_Subset(
        self, vector, indices_vector, n_labels, G, h, A, b, I, B
    ):
        #                            ,
        #                            indexEmpty):
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


if __name__ == "__main__":

    classifer = BaseClassifiers("RF")
    X = [[1, 2, 3], [3, 4, 3], [5, 6, 0]]  # -> X_Train
    Y = [[1, 0], [0, 1], [1, 1]]  # -> Y_Train
    # classifer.fit(X, Y)  # -> train model for X_Train and Y_Train

    predictor = PredictPartialOrder(classifer)

    predictor.fit(X, Y)
    predictor.predict(X, Y, X_test, "subset")

    # predictor.predict(None, None, "subset")
    # predictor.predict(None, None, "hamming")
