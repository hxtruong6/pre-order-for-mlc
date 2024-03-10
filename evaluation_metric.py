import numpy as np
from sklearn.metrics import hamming_loss


class EvaluationMetric:
    def hamming_loss(self, predicted_Y, true_Y):
        return 1 - hamming_loss(predicted_Y, true_Y)

    def f1(self, predicted_Y, true_Y):
        f1 = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            if max(predicted_Y[index]) == 0 and max(true_Y[index]) == 0:
                f1 += 1
            else:
                f1 += (2 * np.dot(predicted_Y[index], true_Y[index])) / (
                    np.sum(predicted_Y[index]) + np.sum(true_Y[index])
                )
        return f1 / n_instances

    def jaccard(self, predicted_Y, true_Y):
        jaccard = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            if max(predicted_Y[index]) == 0 and max(true_Y[index]) == 0:
                jaccard += 1
            else:
                jaccard += (np.dot(predicted_Y[index], true_Y[index])) / (
                    np.sum(predicted_Y[index])
                    + np.sum(true_Y[index])
                    - np.dot(predicted_Y[index], true_Y[index])
                )
        return jaccard / n_instances

    def subset0_1(self, predicted_Y, true_Y):
        subset0_1 = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            if list(predicted_Y[index]) == list(true_Y[index]):
                subset0_1 += 1
        return subset0_1 / n_instances

    def subset_exact_match(self, predicted_Y, true_Y):
        n_instances = len(predicted_Y)
        n_labels = len(predicted_Y[0])
        subset_exact_match = np.zeros(n_labels)

        for index in range(n_instances):
            matched_positions = np.sum(
                [
                    1 if predicted_Y[index][x] == true_Y[index][x] else 0
                    for x in range(n_labels)
                ]
            )
            for pos in range(matched_positions):
                subset_exact_match[pos] += 1
        return [x / n_instances for x in subset_exact_match]

    def recall(self, predicted_Y, true_Y):
        recall = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            if np.dot(predicted_Y[index], true_Y[index]) == np.sum(true_Y[index]):
                recall += 1
        return recall / n_instances
