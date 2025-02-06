import csv
from logging import ERROR, INFO, log, basicConfig
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum
from collections import defaultdict
import pandas as pd
from utils.results_manager import ExperimentResults
from evaluation_metric import EvaluationMetric, EvaluationMetricName

basicConfig(level=INFO)


class OrderType(Enum):
    PRE_ORDER = "PreOrder"
    PARTIAL_ORDER = "PartialOrder"


class MetricType(Enum):
    HAMMING = "Hamming"
    SUBSET = "Subset"


class PredictionType(Enum):
    # PREFERENCE_ORDER = "PreferenceOrder"
    BINARY_VECTOR = "BinaryVector"


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
        PredictionType.BINARY_VECTOR: [
            EvaluationMetricName.HAMMING_ACCURACY,
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

    def load_results(self, dataset_name: str, noisy_rate: float):
        """Load results from pickle file and process predictions."""
        results = ExperimentResults.load_results(
            self.results_dir,
            dataset_name,
            noisy_rate,
        )
        log(INFO, f"Loaded results for {dataset_name} with shape {results.shape}")
        self.df = results

    def evaluate_metric(
        self,
        metric_name: EvaluationMetricName,
        prediction_type: PredictionType,
        predictions: np.ndarray,
        true_labels: np.ndarray,
    ) -> float:
        """Calculate specific evaluation metric."""
        try:
            # print(
            #     f"Evaluating metric: {metric_name} for {prediction_type}: {prediction_type==PredictionType.BINARY_VECTOR} and {metric_name==EvaluationMetricName.HAMMING_ACCURACY}"
            # )
            if prediction_type == PredictionType.BINARY_VECTOR:
                if metric_name == EvaluationMetricName.HAMMING_ACCURACY:
                    print(
                        f"*** Evaluating metric: {metric_name} for {prediction_type}: \nPredictions: \n{predictions} \nTrue labels: \n{true_labels}"
                    )
                    return self.evaluation_metric.hamming_accuracy(
                        predictions, true_labels
                    )
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")
            else:
                raise ValueError(f"Unknown prediction type: {prediction_type}")
        except Exception as e:
            raise Exception(f"Error calculating {metric_name}: {e}")

    def aggregate_results(self, metric_values: List[float]) -> Dict[str, float]:
        """Calculate mean and standard deviation across folds."""
        if not metric_values:
            return {"mean": 0.0, "std": 0.0}
        return {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values)),
        }

    def get_data_df(
        self,
        dataset_name: str,
        base_learner_name: str,
    ):
        """Get data values for a specific dataset, base learner, target metric, preference order, and height."""
        log(
            INFO,
            f"Getting data values for {dataset_name}, {base_learner_name}",
        )
        return self.df[
            (self.df["dataset_name"] == dataset_name)
            & (self.df["base_learner_name"] == base_learner_name)
            # & (self.results_df["target_metric"] == target_metric)
            # & (self.results_df["preference_order"] == preference_order)
            # & (
            #     self.results_df["height"].isna()
            #     if height is None
            #     else self.results_df["height"] == height
            # )
        ]

    def evaluate_dataset(self, dataset_name: str, noisy_rate: float):
        """
        Evaluate results for a specific dataset and noise rate.

        Args:
            dataset_name: Name of the dataset
            noisy_rate: Noise rate used in the experiment
        """
        try:
            log(
                INFO,
                f"Processing results for {dataset_name} with noise rate {noisy_rate}",
            )

            self.eval_results = []

            for base_learner_name in self.df["base_learner_name"].unique():
                log(INFO, f"Processing results for {base_learner_name}")

                # With each evaluation metric, we need to get the data values
                for prediction_type in PredictionType:
                    log(INFO, f"Prediction type: {prediction_type}")

                    # Get the data values
                    data_df = self.get_data_df(
                        dataset_name,
                        base_learner_name,
                    )
                    print(data_df)
                    print(data_df.columns)

                    results = {}
                    # With each inference algorithm, we need to evaluate by repeat_time and fold then aggregate the results
                    for inference_algorithm in EvaluationConfig.INFERENCE_ALGORITHMS:
                        log(INFO, f"Inference algorithm: {inference_algorithm.key}")

                        preference_order = inference_algorithm.order_type.value
                        metric = inference_algorithm.metric.value
                        height = inference_algorithm.height

                        log(INFO, f"Preference order: {preference_order}")
                        log(INFO, f"Metric: {metric}")
                        log(INFO, f"Height: {height}")

                        df1 = data_df[
                            (data_df["preference_order"] == preference_order)
                            & (data_df["target_metric"] == metric)
                            & (
                                data_df["height"] == height
                                if height is not None
                                else data_df["height"].isna()
                            )
                        ]  # [["repeat_time", "fold", "Y_predicted", "Y_test"]]

                        for eval_metric in EvaluationConfig.EVALUATION_METRICS[
                            prediction_type
                        ]:
                            log(INFO, f"Evaluation metric: {eval_metric}")

                            for repeat_time in data_df["repeat_time"].unique():
                                log(INFO, f"Repeat time: {repeat_time}")

                                result_folds = []
                                for fold in data_df["fold"].unique():
                                    log(INFO, f"Fold: {fold}")
                                    df2 = df1[
                                        (df1["repeat_time"] == repeat_time)
                                        & (df1["fold"] == fold)
                                    ]
                                    print("df2", df2)

                                    result_folds.append(
                                        self.evaluate_metric(
                                            eval_metric,
                                            prediction_type,
                                            df2["Y_predicted"].values[0],  # type: ignore
                                            df2["Y_test"].values[0],  # type: ignore
                                        )
                                    )

                            print(result_folds, "result_folds")
                            res = self.aggregate_results(result_folds)
                            # Add the results to the evaluation dataframe
                            self.eval_results.append(
                                {
                                    "Base_Learner": base_learner_name,
                                    "Algorithm": inference_algorithm.key,
                                    "Prediction_Type": prediction_type.value,
                                    "Metric": eval_metric.value,
                                    "Mean": res["mean"],
                                    "Std": res["std"],
                                },
                            )

            log(INFO, f"Evaluation completed for {dataset_name}")

        except Exception as e:
            log(
                ERROR,
                f"Evaluation failed for {dataset_name}: {str(e)}",
                exc_info=True,
            )
            raise

    def _validate_shapes(
        self, predictions: np.ndarray, true_labels: np.ndarray, algo_id: str
    ) -> bool:
        """
        Validate shapes of prediction and true label arrays.

        Args:
            predictions: Predicted labels
            true_labels: True labels
            algo_id: Algorithm identifier for logging

        Returns:
            bool: Whether shapes are valid
        """
        if predictions is None or true_labels is None:
            log(ERROR, f"Missing data for {algo_id}")
            return False

        if predictions.shape != true_labels.shape:
            log(
                ERROR,
                f"Shape mismatch in {algo_id}: {predictions.shape} vs {true_labels.shape}",
            )
            return False

        return True

    def save_results(self, output_path: str):
        """Save evaluation results to CSV and Excel."""
        try:
            # Prepare data for DataFrame
            rows = []
            print("self.eval_results", self.eval_results)

            # Create DataFrame
            df = pd.DataFrame(self.eval_results)

            # Save as CSV
            df.to_csv(output_path + ".csv", index=False)

            # Save as Excel with multiple sheets
            with pd.ExcelWriter(output_path + ".xlsx") as writer:
                # Overall results, get 5 decimal numbers for columns "Mean" and "Std"
                df.to_excel(
                    writer,
                    sheet_name="Overall",
                    index=False,
                    float_format="%.5f",
                )

            log(INFO, f"Results saved to {output_path}.csv and {output_path}.xlsx")

        except Exception as e:
            raise Exception(f"Failed to save results: {str(e)}")


def main():
    # Example usage
    evaluator = EvaluationFramework("./results")

    # Parameters
    dataset_name = "emotions"
    noisy_rate = 0.0

    try:
        evaluator.load_results(dataset_name, noisy_rate)
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
