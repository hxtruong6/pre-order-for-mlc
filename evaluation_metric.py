from enum import Enum
import numpy as np
from sklearn.metrics import hamming_loss


class EvaluationMetricName(Enum):
    HAMMING_ACCURACY = "hamming_accuracy"
    F1 = "f1"
    JACCARD = "jaccard"
    SUBSET0_1 = "subset0_1"
    MEAN_IR = "mean_ir"  # imbalance ratio (for whole dataset)
    CV_IR = "cv_ir"  # coefficient of variation of imbalance ratio (for whole dataset)
    MFRD = "mfrd"  # Maximum False Rate Difference
    AFRD = "afrd"  # Average False Rate Difference
    # Partial Abstention Metrics
    HAMMING_ACCURACY_PA = (
        "hamming_accuracy_pa"  # Hamming accuracy with partial abstention
    )
    SUBSET0_1_PA = "subset0_1_pa"  # Subset accuracy with partial abstention
    F1_PA = "f1_pa"  # F1 score with partial abstention
    # New Abstention Metrics
    AREC = "arec"  # Average Recall per Label
    AABS = "aabs"  # Average Abstention per Label
    REC = "rec"  # Instance-Wise Recall
    ABS = "abs"  # Instance-Wise Abstention
    # SUBSET_EXACT_MATCH = "subset_exact_match"
    # RECALL = "recall"
    # HAMMING_ACCURACY_PL = "hamming_accuracy_PL"
    # SUBSET0_1_PL = "subset0_1_PL"

    HAMMING_ACCURACY_PRE_ORDER = "hamming_accuracy_PRE_ORDER"
    SUBSET0_1_PRE_ORDER = "subset0_1_PRE_ORDER"
    HAMMING_ACCURACY_PARTIAL_ORDER = "hamming_accuracy_PARTIAL_ORDER"
    SUBSET0_1_PARTIAL_ORDER = "subset0_1_PARTIAL_ORDER"


class EvaluationMetric:

    def list_metrics(self):
        return [metric.value for metric in EvaluationMetricName]

    # def calculate(
    #     self,
    #     metric_name: EvaluationMetricName,
    #     predicted_Y: np.ndarray,
    #     true_Y: np.ndarray,
    # ) -> float:
    #     if metric_name == EvaluationMetricName.HAMMING_ACCURACY.value:
    #         return self.hamming_accuracy(predicted_Y, true_Y)
    #     elif metric_name == EvaluationMetricName.F1.value:
    #         return self.f1(predicted_Y, true_Y)
    #     elif metric_name == EvaluationMetricName.SUBSET0_1.value:
    #         return self.subset0_1(predicted_Y, true_Y)
    #     elif metric_name == EvaluationMetricName.MEAN_IR.value:
    #         return self.mean_ir(true_Y)
    #     elif metric_name == EvaluationMetricName.CV_IR.value:
    #         return self.cv_ir(true_Y)
    #     elif metric_name == EvaluationMetricName.MFRD.value:
    #         return self.mfrd(predicted_Y, true_Y)
    #     elif metric_name == EvaluationMetricName.AFRD.value:
    #         return self.afrd(predicted_Y, true_Y)
    #     elif metric_name == EvaluationMetricName.HAMMING_ACCURACY_PA.value:
    #         return self.hamming_accuracy_pa(predicted_Y, true_Y)
    #     elif metric_name == EvaluationMetricName.SUBSET0_1_PA.value:
    #         return self.subset0_1_pa(predicted_Y, true_Y)
    #     elif metric_name == EvaluationMetricName.F1_PA.value:
    #         return self.f1_pa(predicted_Y, true_Y)
    #     elif metric_name == EvaluationMetricName.AREC.value:
    #         return self.arec(predicted_Y, true_Y)
    #     elif metric_name == EvaluationMetricName.AABS.value:
    #         return self.aabs(predicted_Y, true_Y)
    #     elif metric_name == EvaluationMetricName.REC.value:
    #         return self.rec(predicted_Y, true_Y)
    #     elif metric_name == EvaluationMetricName.ABS.value:
    #         return self.abs(predicted_Y, true_Y)
    #     else:
    #         raise ValueError("Invalid metric name")

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

    def hamming_accuracy_PRE_ORDER(self, true_Y, indices_vector, bopos):
        n_labels = len(true_Y[0])
        n_instances = len(true_Y)

        ham_acc_PRE_ORDER = 0
        for index in range(n_instances):
            ham_acc = 0
            for i in range(n_labels - 1):
                for j in range(i + 1, n_labels):
                    if true_Y[index, i] == 1 and true_Y[index, j] == 0:
                        ham_acc += bopos[index, indices_vector[f"{i}_{j}_{0}"]]
                    elif true_Y[index, i] == 0 and true_Y[index, j] == 1:
                        ham_acc += bopos[index, indices_vector[f"{i}_{j}_{1}"]]
                    elif true_Y[index, i] == 0 and true_Y[index, j] == 0:
                        ham_acc += bopos[index, indices_vector[f"{i}_{j}_{2}"]]
                    else:
                        ham_acc += bopos[index, indices_vector[f"{i}_{j}_{3}"]]
            ham_acc_PRE_ORDER += ham_acc / int(n_labels * (n_labels - 1) * 0.5)
        return ham_acc_PRE_ORDER / n_instances

    def subset0_1_accuracy_PRE_ORDER(self, true_Y, indices_vector, bopos):
        subset0_1_PRE_ORDER = 0
        n_instances = len(true_Y)
        for index in range(n_instances):
            current_true_Y = true_Y[index]
            current_bopos = bopos[index]
            subset0_1_PRE_ORDER += self.subset0_1_PRE_ORDER_instance(
                current_true_Y,
                indices_vector,
                current_bopos,
            )
        return subset0_1_PRE_ORDER / n_instances

    def subset0_1_PRE_ORDER_instance(
        self,
        current_true_Y,
        indices_vector,
        bopos,
    ):
        # predicted_partial_orders =
        n_labels = len(current_true_Y)
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                if current_true_Y[i] == 1 and current_true_Y[j] == 0:
                    if bopos[indices_vector[f"{i}_{j}_{0}"]] == 0:
                        return 0
                elif current_true_Y[i] == 0 and current_true_Y[j] == 1:
                    if bopos[indices_vector[f"{i}_{j}_{1}"]] == 0:
                        return 0
                elif current_true_Y[i] == 0 and current_true_Y[j] == 0:
                    if int(bopos[indices_vector[f"{i}_{j}_{2}"]]) == 0:
                        return 0
                else:
                    if bopos[indices_vector[f"{i}_{j}_{3}"]] == 0:
                        return 0
        return 1

    def hamming_accuracy_PARTIAL_ORDER(self, true_Y, indices_vector, bopos):
        n_labels = len(true_Y[0])
        n_instances = len(true_Y)
        ham_acc_PAR_ORDER = 0
        for index in range(n_instances):
            ham_acc = 0
            for i in range(n_labels - 1):
                for j in range(i + 1, n_labels):
                    if true_Y[index, i] == 1 and true_Y[index, j] == 0:
                        ham_acc += bopos[index, indices_vector[f"{i}_{j}_{0}"]]
                    elif true_Y[index, i] == 0 and true_Y[index, j] == 1:
                        ham_acc += bopos[index, indices_vector[f"{i}_{j}_{1}"]]
                    else:
                        ham_acc += bopos[index, indices_vector[f"{i}_{j}_{2}"]]
            ham_acc_PAR_ORDER += ham_acc / int(n_labels * (n_labels - 1) * 0.5)
        return ham_acc_PAR_ORDER / n_instances

    def subset0_1_accuracy_PARTIAL_ORDER(self, true_Y, indices_vector, bopos):
        subset0_1_PAR_ORDER = 0
        n_instances = len(true_Y)
        for index in range(n_instances):
            current_true_Y = true_Y[index]
            current_bopos = bopos[index]
            subset0_1_PAR_ORDER += self.subset0_1_PARTIAL_ORDER_instance(
                current_true_Y,
                indices_vector,
                current_bopos,
            )
        return subset0_1_PAR_ORDER / n_instances

    def subset0_1_PARTIAL_ORDER_instance(
        self,
        current_true_Y,
        indices_vector,
        bopos,
    ):
        n_labels = len(current_true_Y)
        for i in range(n_labels - 1):
            for j in range(i + 1, n_labels):
                if current_true_Y[i] == 1 and current_true_Y[j] == 0:
                    if (bopos[indices_vector[f"{i}_{j}_{0}"]]) == 0:
                        return 0
                elif current_true_Y[i] == 0 and current_true_Y[j] == 1:
                    if (bopos[indices_vector[f"{i}_{j}_{1}"]]) == 0:
                        return 0
                else:
                    if (bopos[indices_vector[f"{i}_{j}_{2}"]]) == 0:
                        return 0
        return 1

    def mean_ir(self, Y: np.ndarray) -> float:
        """
        Calculate Mean Imbalance Ratio (MeanIR) for multi-label dataset.

        Args:
            Y: Binary matrix of shape (n_samples, n_labels) where 1 indicates presence of label

        Returns:
            float: Mean imbalance ratio across all labels
        """
        # Count occurrences of each label
        label_counts = np.sum(Y, axis=0)

        # Find the count of the most frequent label
        max_label_count = np.max(label_counts)

        # Calculate IRperLabel for each label
        IRperLabel = np.zeros(len(label_counts))
        for i, count in enumerate(label_counts):
            if count == 0:
                IRperLabel[i] = np.inf  # Handle case where a label has no instances
            else:
                IRperLabel[i] = max_label_count / count

        # Calculate MeanIR
        return float(np.mean(IRperLabel))

    def cv_ir(self, Y: np.ndarray) -> float:
        """
        Calculate Coefficient of Variation of Imbalance Ratio (CVIR) for multi-label dataset.

        Args:
            Y: Binary matrix of shape (n_samples, n_labels) where 1 indicates presence of label

        Returns:
            float: Coefficient of variation of imbalance ratios
        """
        # Count occurrences of each label
        label_counts = np.sum(Y, axis=0)
        n_labels = len(label_counts)

        # Find the count of the most frequent label
        max_label_count = np.max(label_counts)

        # Calculate IRperLabel for each label
        IRperLabel = np.zeros(n_labels)
        for i, count in enumerate(label_counts):
            if count == 0:
                IRperLabel[i] = np.inf  # Handle case where a label has no instances
            else:
                IRperLabel[i] = max_label_count / count

        # Calculate MeanIR
        MeanIR = np.mean(IRperLabel)

        # Calculate standard deviation of IRperLabel
        if n_labels > 1:
            IRperLabel_sigma = np.sqrt(
                np.sum((IRperLabel - MeanIR) ** 2) / (n_labels - 1)
            )
        else:
            IRperLabel_sigma = 0  # Avoid division by zero if only one label

        # Calculate CVIR
        return float(IRperLabel_sigma / MeanIR if MeanIR != 0 else np.inf)

    def mfrd(self, predicted_Y: np.ndarray, true_Y: np.ndarray) -> float:
        """
        Calculate Maximum False Rate Difference (MFRD) for multi-label classification.

        Args:
            predicted_Y: Binary matrix of shape (n_samples, n_labels) containing predicted labels
            true_Y: Binary matrix of shape (n_samples, n_labels) containing true labels

        Returns:
            float: Maximum absolute difference between FPR and FNR across all labels
        """
        n_labels = predicted_Y.shape[1]
        max_diff = 0.0

        for k in range(n_labels):
            # Calculate TP, FP, TN, FN for current label
            tp = np.sum((predicted_Y[:, k] == 1) & (true_Y[:, k] == 1))
            fp = np.sum((predicted_Y[:, k] == 1) & (true_Y[:, k] == 0))
            tn = np.sum((predicted_Y[:, k] == 0) & (true_Y[:, k] == 0))
            fn = np.sum((predicted_Y[:, k] == 0) & (true_Y[:, k] == 1))

            # Calculate FPR and FNR
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            # Calculate absolute difference
            diff = abs(fpr - fnr)
            max_diff = max(max_diff, diff)

        return float(max_diff)

    def afrd(self, predicted_Y: np.ndarray, true_Y: np.ndarray) -> float:
        """
        Calculate Average False Rate Difference (AFRD) for multi-label classification.

        Args:
            predicted_Y: Binary matrix of shape (n_samples, n_labels) containing predicted labels
            true_Y: Binary matrix of shape (n_samples, n_labels) containing true labels

        Returns:
            float: Average absolute difference between FPR and FNR across all labels
        """
        n_labels = predicted_Y.shape[1]
        total_diff = 0.0

        for k in range(n_labels):
            # Calculate TP, FP, TN, FN for current label
            tp = np.sum((predicted_Y[:, k] == 1) & (true_Y[:, k] == 1))
            fp = np.sum((predicted_Y[:, k] == 1) & (true_Y[:, k] == 0))
            tn = np.sum((predicted_Y[:, k] == 0) & (true_Y[:, k] == 0))
            fn = np.sum((predicted_Y[:, k] == 0) & (true_Y[:, k] == 1))

            # Calculate FPR and FNR
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            # Calculate absolute difference
            diff = abs(fpr - fnr)
            total_diff += diff

        return float(total_diff / n_labels)

    def hamming_accuracy_pa(self, predicted_Y: np.ndarray, true_Y: np.ndarray) -> float:
        """
        Calculate Hamming accuracy with partial abstention.
        In this case, -1 in predicted_Y indicates abstention.

        Args:
            predicted_Y: Binary matrix of shape (n_samples, n_labels) containing predicted labels (-1 for abstention)
            true_Y: Binary matrix of shape (n_samples, n_labels) containing true labels

        Returns:
            float: Hamming accuracy considering partial abstention
        """
        n_samples, n_labels = predicted_Y.shape
        correct_predictions = 0
        total_predictions = 0

        for i in range(n_samples):
            for j in range(n_labels):
                if predicted_Y[i, j] != -1:  # Only count non-abstained predictions
                    if predicted_Y[i, j] == true_Y[i, j]:
                        correct_predictions += 1
                    total_predictions += 1

        return (
            float(correct_predictions / total_predictions)
            if total_predictions > 0
            else 0.0
        )

    def subset0_1_pa(self, predicted_Y: np.ndarray, true_Y: np.ndarray) -> float:
        """
        Calculate Subset accuracy with partial abstention.
        In this case, -1 in predicted_Y indicates abstention.

        Args:
            predicted_Y: Binary matrix of shape (n_samples, n_labels) containing predicted labels (-1 for abstention)
            true_Y: Binary matrix of shape (n_samples, n_labels) containing true labels

        Returns:
            float: Subset accuracy considering partial abstention
        """
        n_samples = len(predicted_Y)
        correct_predictions = 0
        total_predictions = 0

        for i in range(n_samples):
            # Get non-abstained predictions
            non_abstained_mask = predicted_Y[i] != -1
            if np.any(
                non_abstained_mask
            ):  # Only count if there are non-abstained predictions
                if np.array_equal(
                    predicted_Y[i][non_abstained_mask], true_Y[i][non_abstained_mask]
                ):
                    correct_predictions += 1
                total_predictions += 1

        return (
            float(correct_predictions / total_predictions)
            if total_predictions > 0
            else 0.0
        )

    def f1_pa(self, predicted_Y: np.ndarray, true_Y: np.ndarray) -> float:
        """
        Calculate F1 score with partial abstention.
        In this case, -1 in predicted_Y indicates abstention.

        Args:
            predicted_Y: Binary matrix of shape (n_samples, n_labels) containing predicted labels (-1 for abstention)
            true_Y: Binary matrix of shape (n_samples, n_labels) containing true labels

        Returns:
            float: F1 score considering partial abstention
        """
        n_samples = len(predicted_Y)
        f1_sum = 0
        total_instances = 0

        for i in range(n_samples):
            # Get non-abstained predictions
            non_abstained_mask = predicted_Y[i] != -1
            if np.any(
                non_abstained_mask
            ):  # Only count if there are non-abstained predictions
                pred = predicted_Y[i][non_abstained_mask]
                true = true_Y[i][non_abstained_mask]

                if np.sum(pred) == 0 and np.sum(true) == 0:
                    f1_sum += 1
                else:
                    tp = np.sum((pred == 1) & (true == 1))
                    fp = np.sum((pred == 1) & (true == 0))
                    fn = np.sum((pred == 0) & (true == 1))

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0
                        else 0
                    )
                    f1_sum += f1

                total_instances += 1

        return float(f1_sum / total_instances) if total_instances > 0 else 0.0

    def arec(self, predicted_Y: np.ndarray, true_Y: np.ndarray) -> float:
        """
        Compute Average Recall per Label (AREC).

        Args:
            predicted_Y: Predicted label vectors, shape (T, K), with -1 for abstention.
            true_Y: True label vectors, shape (T, K).

        Returns:
            float: AREC score.
        """
        T, K = true_Y.shape

        # Initialize sum for correct predictions
        correct_sum = 0

        # Iterate over instances and labels
        for t in range(T):
            for k in range(K):
                # If prediction is not abstention (-1), check if it matches true label
                # If prediction is abstention (-1), count as correct
                if predicted_Y[t, k] == -1 or predicted_Y[t, k] == true_Y[t, k]:
                    correct_sum += 1

        # Compute AREC
        return float(correct_sum / (K * T))

    def aabs(self, predicted_Y: np.ndarray, true_Y: np.ndarray) -> float:
        """
        Compute Average Abstention per Label (AABS).

        Args:
            predicted_Y: Predicted label vectors, shape (T, K), with -1 for abstention.
            true_Y: True label vectors, shape (T, K).

        Returns:
            float: AABS score.
        """
        T, K = true_Y.shape

        # Count abstentions (-1)
        abstention_sum = np.sum(predicted_Y == -1)

        # Compute AABS
        return float(abstention_sum / (K * T))

    def rec(self, predicted_Y: np.ndarray, true_Y: np.ndarray) -> float:
        """
        Compute Instance-Wise Recall (REC).

        Args:
            predicted_Y: Predicted label vectors, shape (T, K), with -1 for abstention.
            true_Y: True label vectors, shape (T, K).

        Returns:
            float: REC score.
        """
        T = true_Y.shape[0]

        # Initialize sum for correct instances
        correct_instances = 0

        # Iterate over instances
        for t in range(T):
            is_correct = True
            for k in range(true_Y.shape[1]):
                # If prediction is not abstention (-1), it must match true label
                # If prediction is abstention (-1), it is considered correct
                if predicted_Y[t, k] != -1 and predicted_Y[t, k] != true_Y[t, k]:
                    is_correct = False
                    break
            if is_correct:
                correct_instances += 1

        # Compute REC
        return float(correct_instances / T)

    def abs(self, predicted_Y: np.ndarray, true_Y: np.ndarray) -> float:
        """
        Compute Instance-Wise Abstention (ABS).

        Args:
            predicted_Y: Predicted label vectors, shape (T, K), with -1 for abstention.
            true_Y: True label vectors, shape (T, K).

        Returns:
            float: ABS score.
        """
        T = true_Y.shape[0]

        # Sum abstentions per instance
        abstention_sum = np.sum(predicted_Y == -1, axis=1)
        abstention_indices = np.where(abstention_sum = 0, 0, 1)
        # Compute ABS (average over instances)
        return float(np.mean(abstention_indices))
