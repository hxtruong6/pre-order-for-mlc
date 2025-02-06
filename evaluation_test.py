import csv
from logging import ERROR, INFO, log, basicConfig
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum
from collections import defaultdict
import pandas as pd
from utils.results_manager import ExperimentResults
from evaluation_metric import EvaluationMetric

basicConfig(level=INFO)


class OrderType(Enum):
    PRE_ORDER = "Pre_orders"
    # PARTIAL_ORDER = "Partial_orders"


class MetricType(Enum):
    HAMMING = "Hamming"
    # SUBSET = "Subset"


class PredictionType(Enum):
    # PREFERENCE_ORDER = "Preference_order"
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
        # InferenceAlgorithm("IA3", OrderType.PRE_ORDER, MetricType.SUBSET, None),
        # InferenceAlgorithm("IA4", OrderType.PRE_ORDER, MetricType.SUBSET, 2),
        # InferenceAlgorithm("IA5", OrderType.PARTIAL_ORDER, MetricType.HAMMING, None),
        # InferenceAlgorithm("IA6", OrderType.PARTIAL_ORDER, MetricType.HAMMING, 2),
        # InferenceAlgorithm("IA7", OrderType.PARTIAL_ORDER, MetricType.SUBSET, None),
        # InferenceAlgorithm("IA8", OrderType.PARTIAL_ORDER, MetricType.SUBSET, 2),
    ]

    EVALUATION_METRICS = {
        PredictionType.BINARY_VECTOR: [
            "hamming_accuracy",
            # "subset0_1",
            # "f1",
        ],
        # PredictionType.PREFERENCE_ORDER: {
        #     OrderType.PRE_ORDER: [
        #         "hamming_accuracy_PRE_ORDER",
        #         "subset0_1_accuracy_PRE_ORDER",
        #     ],
        #     OrderType.PARTIAL_ORDER: [
        #         "hamming_accuracy_PARTIAL_ORDER",
        #         "subset0_1_accuracy_PARTIAL_ORDER",
        #     ],
        # },
    }


class EvaluationFramework:
    """Framework for evaluating multi-label classification results."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.evaluation_metric = EvaluationMetric()

    def load_results(self, dataset_name: str, noisy_rate: float) -> pd.DataFrame:
        """Load results from pickle file and process predictions."""
        results = ExperimentResults.load_results(
            self.results_dir,
            dataset_name,
            noisy_rate,
        )
        log(INFO, f"Loaded results for {dataset_name} with shape {results.shape}")
        return results

    def evaluate_metric(
        self, metric_name: str, predictions: np.ndarray, true_labels: np.ndarray
    ) -> float:
        """Calculate specific evaluation metric."""
        try:
            if metric_name.startswith("hamming"):
                return self.evaluation_metric.hamming_accuracy(true_labels, predictions)
            # elif metric_name.startswith("subset"):
            #     return self.evaluation_metric.subset0_1(true_labels, predictions)
            # elif metric_name == "f1":
            #     return self.evaluation_metric.f1(true_labels, predictions)
            else:
                log(ERROR, f"Unknown metric: {metric_name}")
                raise ValueError(f"Unknown metric: {metric_name}")
        except Exception as e:
            log(ERROR, f"Error calculating {metric_name}: {str(e)}")
            raise Exception(f"Error calculating {metric_name}: {str(e)}")

    def aggregate_fold_results(self, metric_values: List[float]) -> Dict[str, float]:
        """Calculate mean and standard deviation across folds."""
        if not metric_values:
            return {"mean": 0.0, "std": 0.0}
        return {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values)),
        }

    def evaluate_dataset(self, dataset_name: str, noisy_rate: float):
        """Evaluate results for a specific dataset and noise rate."""
        try:
            results_df = self.load_results(dataset_name, noisy_rate)

            for algo in EvaluationConfig.INFERENCE_ALGORITHMS:
                log(INFO, f"Evaluating algorithm: {algo.id}")

                for pred_type in PredictionType:
                    # metrics = (
                    #     EvaluationConfig.EVALUATION_METRICS[pred_type]
                    #     if pred_type == PredictionType.BINARY_VECTOR
                    #     else EvaluationConfig.EVALUATION_METRICS[pred_type][
                    #         algo.order_type
                    #     ]
                    # )
                    log(INFO, f"PredictionType: {pred_type}")
                    metrics = EvaluationConfig.EVALUATION_METRICS[pred_type]
                    fold_results = defaultdict(list)

                    log(
                        INFO,
                        f"Evaluating {pred_type.value} for {algo.id} with metrics: {metrics}",
                    )

                    # In this step, it should be get all fold results for each algorithm
                    # and then aggregate the results
                    # Using group by [target_metric, preference_order, height] to get the results
                    # with each [target_metric, preference_order, height] will have a list of results of each fold

                    # Process each fold's results
                    for _, fold_data in results_df.iterrows():
                        predictions = fold_data["Y_predicted"]
                        true_labels = fold_data["Y_test"]

                        # Skip if shapes don't match
                        if predictions.shape != true_labels.shape:
                            log(
                                ERROR,
                                f"Shape mismatch in {algo.id}: {predictions.shape} vs {true_labels.shape}",
                            )
                            continue

                        for metric in metrics:
                            value = self.evaluate_metric(
                                metric, predictions, true_labels
                            )
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

            log(INFO, f"Evaluation completed for {dataset_name}")

        except Exception as e:
            log(ERROR, f"Evaluation failed for {dataset_name}: {str(e)}", exc_info=True)
            raise

    def save_results(self, output_path: str):
        """Save evaluation results to CSV and Excel."""
        try:
            # Prepare data for DataFrame
            rows = []
            for algo_key, pred_types in self.results.items():
                for pred_type, metrics in pred_types.items():
                    for metric, stats in metrics.items():
                        rows.append(
                            {
                                "Algorithm": algo_key,
                                "Prediction_Type": pred_type,
                                "Metric": metric,
                                "Mean": stats["mean"],
                                "Std": stats["std"],
                            }
                        )

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Save as CSV
            df.to_csv(output_path + ".csv", index=False)

            # Save as Excel with multiple sheets
            with pd.ExcelWriter(output_path + ".xlsx") as writer:
                # Overall results
                df.to_excel(writer, sheet_name="Overall", index=False)

                # Separate sheets for each prediction type
                for pred_type in PredictionType:
                    pred_df = df[df["Prediction_Type"] == pred_type.value]
                    pred_df.to_excel(writer, sheet_name=pred_type.value, index=False)

            log(INFO, f"Results saved to {output_path}.csv and {output_path}.xlsx")

        except Exception as e:
            log(ERROR, f"Failed to save results: {str(e)}")
            raise


def main():
    # Example usage
    evaluator = EvaluationFramework("./results")

    # Parameters
    dataset_name = "emotions"
    noisy_rate = 0.0

    try:
        # Evaluate dataset
        evaluator.evaluate_dataset(dataset_name, noisy_rate)

        # Save results
        output_base = f"./results/evaluation_{dataset_name}_noisy_{noisy_rate}"
        evaluator.save_results(output_base)

        log(INFO, f"Evaluation completed successfully for {dataset_name}")

    except Exception as e:
        log(ERROR, f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    main()
