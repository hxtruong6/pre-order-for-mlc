import numpy as np

# we shall add the option to predict bipartite graphs by enforcing 
# more constraints in _partialorders_compute_parameters and _preorders_compute_parameters

class predict_partial_order_subset:
    MAX_HEIGHT = 2 # for case with height = 2
    def __init__(self, n_labels, base_learner: str):
        self.base_learner = base_learner

    # call pairwise probabilistic information
    # encode parameter and target function 
    # find the Bayes-optimal partial order with respect to (w.r.t) the subset 0/1 accuracy
    # with the option of restricting the height to at most 2
    #  subset| weighted_Hamming

    def _predict_PaOr_Subset(self, X_test, pairwise_3classifier, height=None):
#    def _predict_PaOr_Subset(self, pairwise_probability_information, height=None):

        indices_vector = {}
        indVec = 0
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                for l in range(3):
                    key = "%i_%i_%i" % (i, j, l)
                    indices_vector[key] = indVec
                    indVec += 1
        G, h, A, b, I, B = self._encode_parameters_PaOr_Subset(
            base_learner, indices_vector, n_labels, height
        )
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

    def _encode_parameters_PaOr_Subset(self, indices_vector, n_labels, height=None):
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

    # call pairwise probabilistic information
    # encode parameter and target function 
    # find the Bayes-optimal partial order w.r.t the hamming accuracy 
    # with the option of restricting the height to at most 2


    

class predict_pre_order:
    def __init__(self, n_labels, base_learner: str):
        self.base_learner = base_learner

    def save_model(model, filename):
        import pickle

        with open(filename, "wb") as f:
            pickle.dump(model, f)
    def _predict_PreOr_Subset(self, X_test, pairwise_4classifier):
        indices_vector = {}
        indVec = 0
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                for l in range(4):
                    key = "%i_%i_%i" % (i, j, l)
                    indices_vector[key] = indVec
                    indVec += 1
        G, h, A, b, I, B = MLCPredictor._preorders_compute_parameters(
            base_learner, indices_vector, n_labels
        )
        predicted_Y = []
        predicted_preorders = []
        n_instances, _ = X_test.shape
        for index in range(n_instances):
            #            print(index, n_instances)
            vector = []
            #           indexEmpty = []
            for i in range(n_labels - 1):
                local_classifier = pairwise_4classifier[i]
                for k_2 in range(n_labels - i - 1):
                    #                for j in range(i+1,n_labels):
                    clf = local_classifier[k_2]
                    j = i + k_2 + 1
                    pairInfor_ori = clf.predict_proba(X_test[index].reshape(1, -1))
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
                    #                    if min(pairInfor) == 0 and max(pairInfor) == 0:
                    #                        print(pairInfor_ori)
                    #                        print(pairInfor)
                    #                        print("aaaaaaaaaaaaaaaaaaa")
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
            #                   empty = [indices_vector["%i_%i_%i"%(i,j,l)] for l in range(4) if pairInfor[l] == 0]
            #                   indexEmpty += empty
            Gtest = np.array(G)
            Atest = np.array(A)
            #            for indCol in indexEmpty:
            #                Gtest[:, indCol] = 0
            #                Atest[:, indCol] = 0
            hard_prediction_indices, predicted_preorder = (
                MLCPredictor._preorders_reasoning_procedure(
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

    def _encode_parameters_PreOr_Subset(self, indices_vector, n_labels):
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

    def _reasoning_procedure_PreOr_Subset(
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

class baselines:
    # Conventional classifiers, which are especially designed to predict binary vectors can be added here
    # We currently include calibarated label ranking (CLR) and ensemble of classifier chains (ECCs)
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
 
    def _ECC(self, X_test):
        pass

    pass