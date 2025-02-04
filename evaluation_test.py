import csv
from logging import ERROR, INFO, log, basicConfig
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum
from collections import defaultdict

basicConfig(level=INFO)


class OrderType(Enum):
    PRE_ORDER = "Pre_orders"
    PARTIAL_ORDER = "Partial_orders"


class MetricType(Enum):
    HAMMING = "Hamming"
    SUBSET = "Subset"


class PredictionType(Enum):
    PREFERENCE_ORDER = "Preference_order"
    BINARY_VECTOR = "Binary_vector"


@dataclass
class InferenceAlgorithm:
    id: str
    order_type: OrderType
    metric: MetricType
    height: int | None

    @property
    def key(self) -> str:
        return f"{self.order_type.value}__{self.metric.value}__{self.height}"


class EvaluationConfig:
    INFERENCE_ALGORITHMS = [
        InferenceAlgorithm("IA1", OrderType.PRE_ORDER, MetricType.HAMMING, None),
        InferenceAlgorithm("IA2", OrderType.PRE_ORDER, MetricType.HAMMING, 2),
        InferenceAlgorithm("IA3", OrderType.PRE_ORDER, MetricType.SUBSET, None),
        InferenceAlgorithm("IA4", OrderType.PRE_ORDER, MetricType.SUBSET, 2),
        InferenceAlgorithm("IA5", OrderType.PARTIAL_ORDER, MetricType.HAMMING, None),
        InferenceAlgorithm("IA6", OrderType.PARTIAL_ORDER, MetricType.HAMMING, 2),
        InferenceAlgorithm("IA7", OrderType.PARTIAL_ORDER, MetricType.SUBSET, None),
        InferenceAlgorithm("IA8", OrderType.PARTIAL_ORDER, MetricType.SUBSET, 2),
    ]

    EVALUATION_METRICS = {
        PredictionType.BINARY_VECTOR: ["hamming_accuracy", "subset0_1", "f1"],
        PredictionType.PREFERENCE_ORDER: {
            OrderType.PRE_ORDER: [
                "hamming_accuracy_PRE_ORDER",
                "subset0_1_accuracy_PRE_ORDER",
            ],
            OrderType.PARTIAL_ORDER: [
                "hamming_accuracy_PARTIAL_ORDER",
                "subset0_1_accuracy_PARTIAL_ORDER",
            ],
        },
    }


class EvaluationFramework:
    """Framework for evaluating multi-label classification results."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    def load_results(self, dataset_name: str, noisy_rate: float) -> Dict:
        """Load results from pickle file."""
        filepath = self.results_dir / f"dataset_{dataset_name}_noisy_{noisy_rate}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        with open(filepath, "rb") as f:
            return pickle.load(f)

    def evaluate_metric(
        self, metric_name: str, predictions: np.ndarray, true_labels: np.ndarray
    ) -> float:
        """Calculate specific evaluation metric."""
        # TODO: call EvaluationMetric with correct metric name
        pass

    def aggregate_fold_results(self, metric_values: List[float]) -> Dict[str, float]:
        """Calculate mean and standard deviation across folds."""
        return {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values)),
        }

    def evaluate_dataset(self, dataset_name: str, noisy_rate: float):
        """Evaluate results for a specific dataset and noise rate."""
        results = self.load_results(dataset_name, noisy_rate)

        for algo in EvaluationConfig.INFERENCE_ALGORITHMS:
            for pred_type in PredictionType:
                metrics = (
                    EvaluationConfig.EVALUATION_METRICS[pred_type]
                    if pred_type == PredictionType.BINARY_VECTOR
                    else EvaluationConfig.EVALUATION_METRICS[pred_type][algo.order_type]
                )

                fold_results = defaultdict(list)

                # Process each fold's results
                for fold_data in results:
                    predictions = fold_data[f"predictions_{algo.key}"]
                    true_labels = fold_data["Y_test"]

                    for metric in metrics:
                        value = self.evaluate_metric(metric, predictions, true_labels)
                        fold_results[metric].append(value)

                # Aggregate results across folds
                aggregated_results = {
                    metric: self.aggregate_fold_results(values)
                    for metric, values in fold_results.items()
                }

                # Store results
                if algo.key not in self.results:
                    self.results[algo.key] = {}
                self.results[algo.key][pred_type.value] = aggregated_results

    def save_results(self, output_path: str):
        """Save evaluation results to CSV."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Algorithm", "Prediction_Type", "Metric", "Mean", "Std"])

            for algo_key, pred_types in self.results.items():
                for pred_type, metrics in pred_types.items():
                    for metric, stats in metrics.items():
                        writer.writerow(
                            [algo_key, pred_type, metric, stats["mean"], stats["std"]]
                        )


def main():
    # Example usage
    evaluator = EvaluationFramework("./results")

    # Evaluate specific dataset
    dataset_name = "emotions"
    noisy_rate = 0.0

    try:
        evaluator.evaluate_dataset(dataset_name, noisy_rate)
        evaluator.save_results(f"evaluation_{dataset_name}_{noisy_rate}.csv")
        log(INFO, f"Evaluation completed successfully for {dataset_name}")
    except Exception as e:
        log(ERROR, f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    main()
