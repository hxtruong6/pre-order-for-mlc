from enum import Enum
import numpy as np
from sklearn.metrics import hamming_loss


class EvaluationMetricName(Enum):
    HAMMING_ACCURACY = "hamming_accuracy"
    F1 = "f1"
    JACCARD = "jaccard"
    SUBSET0_1 = "subset0_1"
    SUBSET_EXACT_MATCH = "subset_exact_match"
    RECALL = "recall"
    HAMMING_ACCURACY_PL = "hamming_accuracy_PL"
    SUBSET0_1_PL = "subset0_1_PL"


class EvaluationMetric:

    def list_metrics(self):
        return [metric.value for metric in EvaluationMetricName]

    def calculate(
        self,
        metric_name: EvaluationMetricName,
        predicted_Y: np.ndarray,
        true_Y: np.ndarray,
    ) -> float:
        if metric_name == EvaluationMetricName.HAMMING_ACCURACY.value:
            return self.hamming_accuracy(predicted_Y, true_Y)
        # elif metric_name == EvaluationMetricName.F1.value:
        #     return self.f1(predicted_Y, true_Y)
        # elif metric_name == EvaluationMetricName.JACCARD.value:
        #     return self.jaccard(predicted_Y, true_Y)
        # elif metric_name == EvaluationMetricName.SUBSET0_1.value:
        #     return self.subset0_1(predicted_Y, true_Y)
        # elif metric_name == EvaluationMetricName.SUBSET_EXACT_MATCH.value:
        #     return self.subset_exact_match(predicted_Y, true_Y)
        # elif metric_name == EvaluationMetricName.RECALL.value:
        #     return self.recall(predicted_Y, true_Y)
        # elif metric_name == EvaluationMetricName.HAMMING_ACCURACY_PL.value:
        #     return self.hamming_accuracy_PL(predicted_Y, true_Y)
        # elif metric_name == EvaluationMetricName.SUBSET0_1_PL.value:
        #     return self.subset0_1_PL(predicted_Y, true_Y)
        else:
            raise ValueError("Invalid metric name")

    def hamming_accuracy(self, predicted_Y, true_Y) -> float:
        return 1 - hamming_loss(predicted_Y, true_Y)  # type: ignore

    def f1(self, predicted_Y: np.ndarray, true_Y: np.ndarray) -> float:
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

    def hamming_accuracy_PRE_ORDER(self, predicted_preorders, true_Y, indices_vector):
        n_labels = len(true_Y[0])

        ham_acc_PRE_ORDER = 0
        n_instances = len(predicted_preorders)
        for index in range(n_instances):
            current_predicted_preorder = predicted_preorders[index]
            ham_acc = 0
            for i in range(n_labels - 1):
                for j in range(i + 1, n_labels):
                    if true_Y[index, i] == 1 and true_Y[index, j] == 0:
                        ham_acc += current_predicted_preorder[
                            indices_vector[f"{i}_{j}_{0}"]
                        ]
                    elif true_Y[index, i] == 0 and true_Y[index, j] == 1:
                        ham_acc += current_predicted_preorder[
                            indices_vector[f"{i}_{j}_{1}"]
                        ]
                    elif true_Y[index, i] == 0 and true_Y[index, j] == 0:
                        ham_acc += current_predicted_preorder[
                            indices_vector[f"{i}_{j}_{2}"]
                        ]
                    else:
                        ham_acc += current_predicted_preorder[
                            indices_vector[f"{i}_{j}_{3}"]
                        ]
            ham_acc_PRE_ORDER += ham_acc / int(n_labels * (n_labels - 1) * 0.5)
        return ham_acc_PRE_ORDER / n_instances

    def subset0_1_accuracy_PRE_ORDER(self, predicted_preorders, true_Y, indices_vector):
        subset0_1_PRE_ORDER = 0
        n_instances = len(predicted_preorders)
        for index in range(n_instances):
            current_predicted_preorder = predicted_preorders[index]
            current_true_Y = true_Y[index]
            subset0_1_PRE_ORDER += self.subset0_1_PRE_ORDER_instance(
                current_predicted_preorder, current_true_Y, indices_vector
            )
        return subset0_1_PRE_ORDER / n_instances

    def subset0_1_PRE_ORDER_instance(
        self, current_predicted_preorder, current_true_Y, indices_vector
    ):
        n_labels = len(current_true_Y)
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                if current_true_Y[i] == 1 and current_true_Y[j] == 0:
                    if current_predicted_preorder[indices_vector[f"{i}_{j}_{0}"]] == 0:
                        return 0
                elif current_true_Y[i] == 0 and current_true_Y[j] == 1:
                    if current_predicted_preorder[indices_vector[f"{i}_{j}_{1}"]] == 0:
                        return 0
                elif current_true_Y[i] == 0 and current_true_Y[j] == 0:
                    if current_predicted_preorder[indices_vector[f"{i}_{j}_{2}"]] == 0:
                        return 0
                else:
                    if current_predicted_preorder[indices_vector[f"{i}_{j}_{3}"]] == 0:
                        return 0
        return 1

    def hamming_accuracy_PAR_ORDER(
        self, predicted_partialorders, true_Y, indices_vector
    ):
        n_labels = len(true_Y[0])
        ham_acc_PAR_ORDER = 0
        n_instances = len(predicted_partialorders)
        for index in range(n_instances):
            current_predicted_partialorders = predicted_partialorders[index]
            ham_acc = 0
            for i in range(n_labels - 1):
                for j in range(i + 1, n_labels):
                    if true_Y[index, i] == 1 and true_Y[index, j] == 0:
                        ham_acc += current_predicted_partialorders[
                            indices_vector[f"{i}_{j}_{0}"]
                        ]
                    elif true_Y[index, i] == 0 and true_Y[index, j] == 1:
                        ham_acc += current_predicted_partialorders[
                            indices_vector[f"{i}_{j}_{1}"]
                        ]
                    else:
                        ham_acc += current_predicted_partialorders[
                            indices_vector[f"{i}_{j}_{2}"]
                        ]
            ham_acc_PAR_ORDER += ham_acc / int(n_labels * (n_labels - 1) * 0.5)
        return ham_acc_PAR_ORDER / n_instances

    def subset0_1_accuracy_PAR_ORDER(
        self, predicted_partialorders, true_Y, indices_vector
    ):
        subset0_1_PAR_ORDER = 0
        n_instances = len(predicted_partialorders)
        for index in range(n_instances):
            current_predicted_partialorders = predicted_partialorders[index]
            current_true_Y = true_Y[index]
            subset0_1_PAR_ORDER += self.subset0_1_PAR_ORDER_instance(
                current_predicted_partialorders, current_true_Y, indices_vector
            )
        return subset0_1_PAR_ORDER / n_instances

    def subset0_1_PAR_ORDER_instance(
        self, current_predicted_partialorders, current_true_Y, indices_vector
    ):
        n_labels = len(current_true_Y)
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                if current_true_Y[i] == 1 and current_true_Y[j] == 0:
                    if (
                        current_predicted_partialorders[indices_vector[f"{i}_{j}_{0}"]]
                        == 0
                    ):
                        return 0
                elif current_true_Y[i] == 0 and current_true_Y[j] == 1:
                    if (
                        current_predicted_partialorders[indices_vector[f"{i}_{j}_{1}"]]
                        == 0
                    ):
                        return 0
                else:
                    if (
                        current_predicted_partialorders[indices_vector[f"{i}_{j}_{2}"]]
                        == 0
                    ):
                        return 0
        return 1
