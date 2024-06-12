class search_BOPreOs:
    def __init__(self, pairwise_probabilistic_predictions):
    self.pairwise_probabilistic_predictions = pairwise_probabilistic_predictions

    def PRE_ORDER_HAM(self.pairwise_probabilistic_predictions):
        pass

    def BIPARTITE_PRE_ORDER_HAM(self.pairwise_probabilistic_predictions):
        pass

    def PRE_ORDER_SUB(self.pairwise_probabilistic_predictions):
        pass

    def BIPARTITE_PRE_ORDER_SUB(self.pairwise_probabilistic_predictions):
        pass


class search_BOParOs:
    def __init__(self, pairwise_probabilistic_predictions):
    self.pairwise_probabilistic_predictions = pairwise_probabilistic_predictions


    def PARTIAL_ORDER_HAM(self.pairwise_probabilistic_predictions):
        pass

    def BIPARTITE_PARTIAL_ORDER_HAM(self.pairwise_probabilistic_predictions):
        pass

    def PARTIAL_ORDER_SUB(self.pairwise_probabilistic_predictions):
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
        for n in range(n_instances):
            vector = []
            for i in range(n_labels - 1):
                for j in range(i + 1, self.n_labels):
                    pairInfor = [-np.log(pairwise_probabilistic_predictions[f"{i}_{j}_{n}_{l}"]) for l in range(3)]             
                    vector += pairInfor
            Gtest = np.array(G)
            Atest = np.array(A)
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
            hard_prediction = [
                1 if x in hard_prediction_indices else 0 for x in range(n_labels)
            ]
            predicted_Y.append(hard_prediction)
            predicted_preorders.append(predicted_parorders)
        return predicted_Y, predicted_parorders

    def BIPARTITE_PARTIAL_ORDER_SUB(self.pairwise_probabilsitic_predictions):
        pass