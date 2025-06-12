import argparse
import csv
import json
from logging import ERROR, INFO, log, basicConfig
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
from collections import defaultdict
import pandas as pd
from config import ConfigManager
from utils.results_manager import ExperimentResults
from evaluation_metric import EvaluationMetric, EvaluationMetricName

# basicConfig(level=INFO)
basicConfig(level=ERROR)


class OrderType(Enum):
    PRE_ORDER = "PreOrder"
    PARTIAL_ORDER = "PartialOrder"


class MetricType(Enum):
    HAMMING = "Hamming"
    SUBSET = "Subset"


class PredictionType(Enum):
    BINARY_VECTOR = "BinaryVector"
    PREFERENCE_ORDER = "PreferenceOrder"


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
            EvaluationMetricName.SUBSET0_1,
            EvaluationMetricName.F1,
            EvaluationMetricName.MEAN_IR,
            EvaluationMetricName.CV_IR,
            EvaluationMetricName.MFRD,
            EvaluationMetricName.AFRD,
            # Partial Abstention Metrics
            EvaluationMetricName.HAMMING_ACCURACY_PA,
            EvaluationMetricName.SUBSET0_1_PA,
            EvaluationMetricName.F1_PA,
            # New Abstention Metrics
            EvaluationMetricName.AREC,
            EvaluationMetricName.AABS,
            EvaluationMetricName.REC,
            EvaluationMetricName.ABS,
        ],
        PredictionType.PREFERENCE_ORDER: {
            OrderType.PRE_ORDER: [
                EvaluationMetricName.HAMMING_ACCURACY_PRE_ORDER,
                EvaluationMetricName.SUBSET0_1_PRE_ORDER,
            ],
            OrderType.PARTIAL_ORDER: [
                EvaluationMetricName.HAMMING_ACCURACY_PARTIAL_ORDER,
                EvaluationMetricName.SUBSET0_1_PARTIAL_ORDER,
            ],
        },
    }

    # Metrics that should be calculated on the entire dataset, not per fold
    DATASET_LEVEL_METRICS = {
        EvaluationMetricName.MEAN_IR,
        EvaluationMetricName.CV_IR,
    }


class EvaluationFramework:
    """Framework for evaluating multi-label classification results."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.evaluation_metric = EvaluationMetric()
        self.df = None

    def load_results(
        self,
        dataset_name: str,
        noisy_rate: float,
        is_clr: bool = False,
        is_br: bool = False,
        is_cc: bool = False,
    ):
        """Load results from file"""
        try:
            self.df = ExperimentResults.load_results(
                self.results_dir,
                dataset_name,
                noisy_rate,
                is_clr=is_clr,
                is_br=is_br,
                is_cc=is_cc,
            )
        except Exception as e:
            log(ERROR, f"Failed to load results: {str(e)}")
            raise

    def evaluate_metric(
        self,
        metric_name: EvaluationMetricName,
        prediction_type: Optional[PredictionType],
        predictions: np.ndarray,
        true_labels: np.ndarray,
        indices_vector: Optional[np.ndarray],
        bopos: Optional[np.ndarray],
        order_type: Optional[OrderType] = None,
        is_clr: bool = False,  # clr: calibrate classifier ranking
        is_br: bool = False,  # binary relevant
        is_cc: bool = False,  # classifier chain
    ) -> float:
        """Calculate specific evaluation metric."""
        try:
            if is_br or is_cc or is_clr:
                return self._evaluate_simple_metric(
                    metric_name, predictions, true_labels, is_clr, is_br, is_cc
                )

            if prediction_type == PredictionType.BINARY_VECTOR:
                return self._evaluate_binary_vector(
                    metric_name, predictions, true_labels
                )
            elif prediction_type == PredictionType.PREFERENCE_ORDER:
                return self._evaluate_preference_order(
                    metric_name,
                    predictions,
                    true_labels,
                    indices_vector,
                    bopos,
                    order_type,
                )
            else:
                raise ValueError(f"Unknown prediction type: {prediction_type}")
        except Exception as e:
            raise Exception(f"Error calculating {metric_name}: {e}")

    def _evaluate_simple_metric(
        self,
        metric_name: EvaluationMetricName,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        is_clr: bool,
        is_br: bool,
        is_cc: bool,
    ) -> float:
        """Evaluate metrics for simple classifiers (BR, CC, CLR)"""
        if metric_name == EvaluationMetricName.HAMMING_ACCURACY:
            return self.evaluation_metric.hamming_accuracy(predictions, true_labels)
        elif metric_name == EvaluationMetricName.SUBSET0_1:
            return self.evaluation_metric.subset0_1(predictions, true_labels)
        elif metric_name == EvaluationMetricName.F1 and (is_clr or is_cc or is_br):
            return self.evaluation_metric.f1(predictions, true_labels)
        elif metric_name == EvaluationMetricName.MEAN_IR:
            return self.evaluation_metric.mean_ir(true_labels)
        elif metric_name == EvaluationMetricName.CV_IR:
            return self.evaluation_metric.cv_ir(true_labels)
        elif metric_name == EvaluationMetricName.MFRD:
            return self.evaluation_metric.mfrd(predictions, true_labels)
        elif metric_name == EvaluationMetricName.AFRD:
            return self.evaluation_metric.afrd(predictions, true_labels)
        elif metric_name == EvaluationMetricName.HAMMING_ACCURACY_PA:
            return self.evaluation_metric.hamming_accuracy_pa(predictions, true_labels)
        elif metric_name == EvaluationMetricName.SUBSET0_1_PA:
            return self.evaluation_metric.subset0_1_pa(predictions, true_labels)
        elif metric_name == EvaluationMetricName.F1_PA:
            return self.evaluation_metric.f1_pa(predictions, true_labels)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def _evaluate_binary_vector(
        self,
        metric_name: EvaluationMetricName,
        predictions: np.ndarray,
        true_labels: np.ndarray,
    ) -> float:
        """Evaluate metrics for binary vector predictions"""
        if metric_name == EvaluationMetricName.HAMMING_ACCURACY:
            return self.evaluation_metric.hamming_accuracy(predictions, true_labels)
        elif metric_name == EvaluationMetricName.SUBSET0_1:
            return self.evaluation_metric.subset0_1(predictions, true_labels)
        elif metric_name == EvaluationMetricName.F1:
            return self.evaluation_metric.f1(predictions, true_labels)
        elif metric_name == EvaluationMetricName.MEAN_IR:
            return self.evaluation_metric.mean_ir(true_labels)
        elif metric_name == EvaluationMetricName.CV_IR:
            return self.evaluation_metric.cv_ir(true_labels)
        elif metric_name == EvaluationMetricName.MFRD:
            return self.evaluation_metric.mfrd(predictions, true_labels)
        elif metric_name == EvaluationMetricName.AFRD:
            return self.evaluation_metric.afrd(predictions, true_labels)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def _evaluate_preference_order(
        self,
        metric_name: EvaluationMetricName,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        indices_vector: Optional[np.ndarray],
        bopos: Optional[np.ndarray],
        order_type: Optional[OrderType],
    ) -> float:
        """Evaluate metrics for preference order predictions"""
        if order_type == OrderType.PRE_ORDER:
            if metric_name == EvaluationMetricName.HAMMING_ACCURACY_PRE_ORDER:
                return self.evaluation_metric.hamming_accuracy_PRE_ORDER(
                    true_labels, indices_vector, bopos
                )
            elif metric_name == EvaluationMetricName.SUBSET0_1_PRE_ORDER:
                return self.evaluation_metric.subset0_1_accuracy_PRE_ORDER(
                    true_labels, indices_vector, bopos
                )
        elif order_type == OrderType.PARTIAL_ORDER:
            if metric_name == EvaluationMetricName.HAMMING_ACCURACY_PARTIAL_ORDER:
                return self.evaluation_metric.hamming_accuracy_PARTIAL_ORDER(
                    true_labels, indices_vector, bopos
                )
            elif metric_name == EvaluationMetricName.SUBSET0_1_PARTIAL_ORDER:
                return self.evaluation_metric.subset0_1_accuracy_PARTIAL_ORDER(
                    true_labels, indices_vector, bopos
                )
        raise ValueError(f"Unknown metric or order type: {metric_name}, {order_type}")

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
    ) -> pd.DataFrame:
        """Get data values for a specific dataset and base learner."""
        if self.df is None:
            log(ERROR, "No data loaded. Call load_results() first.")
            return pd.DataFrame()

        log(
            INFO,
            f"Getting data values for {dataset_name}, {base_learner_name}",
        )

        filtered_df = self.df[
            (self.df["dataset_name"].str.lower() == dataset_name.lower())
            & (self.df["base_learner_name"].str.lower() == base_learner_name.lower())
        ]

        if filtered_df.empty:
            log(
                ERROR,
                f"No data found for dataset {dataset_name} and base learner {base_learner_name}",
            )
            return pd.DataFrame()

        return filtered_df

    def _evaluate_fold(
        self,
        data_df: pd.DataFrame,
        df1: pd.DataFrame,
        eval_metric: EvaluationMetricName,
        prediction_type: Optional[PredictionType],
        order_type: Optional[OrderType] = None,
        is_clr: bool = False,
        is_br: bool = False,
        is_cc: bool = False,
    ) -> List[float]:
        """Evaluate a single fold of results."""
        if data_df.empty or df1.empty:
            log(ERROR, "Empty DataFrame provided for evaluation")
            return []

        result_folds = []
        for repeat_time in data_df["repeat_time"].unique():
            log(INFO, f"Repeat time: {repeat_time}")

            for fold in data_df["fold"].unique():
                log(INFO, f"Fold: {fold}")
                df2 = df1[(df1["repeat_time"] == repeat_time) & (df1["fold"] == fold)]

                if df2.empty:
                    log(
                        ERROR,
                        f"No data found for repeat_time {repeat_time} and fold {fold}",
                    )
                    continue

                try:
                    # Extract values with type checking
                    y_pred = df2["Y_predicted"].values[0]
                    y_test = df2["Y_test"].values[0]
                    indices = (
                        df2["indices_vector"].values[0]
                        if "indices_vector" in df2.columns
                        else None
                    )
                    bopos = (
                        df2["Y_BOPOs"].values[0] if "Y_BOPOs" in df2.columns else None
                    )

                    if not isinstance(y_pred, np.ndarray) or not isinstance(
                        y_test, np.ndarray
                    ):
                        log(
                            ERROR,
                            f"Invalid prediction or test data type for fold {fold}",
                        )
                        continue

                    result_folds.append(
                        self.evaluate_metric(
                            metric_name=eval_metric,
                            prediction_type=prediction_type,
                            predictions=y_pred,
                            true_labels=y_test,
                            indices_vector=indices,
                            bopos=bopos,
                            order_type=order_type,
                            is_clr=is_clr,
                            is_br=is_br,
                            is_cc=is_cc,
                        )
                    )
                except (IndexError, KeyError) as e:
                    log(ERROR, f"Error accessing data for fold {fold}: {str(e)}")
                    continue

        return result_folds

    def evaluate_dataset(self, dataset_name: str, noisy_rate: float):
        """Evaluate results for a specific dataset and noise rate."""
        try:
            if self.df is None:
                log(ERROR, "No data loaded. Call load_results() first.")
                return

            log(
                INFO,
                f"Processing results for {dataset_name} with noise rate {noisy_rate}",
            )

            self.eval_results = []

            for base_learner_name in self.df["base_learner_name"].unique():
                log(INFO, f"Processing results for {base_learner_name}")
                data_df = self.get_data_df(dataset_name, base_learner_name)

                for prediction_type in PredictionType:
                    log(INFO, f"Prediction type: {prediction_type}")

                    for inference_algorithm in EvaluationConfig.INFERENCE_ALGORITHMS:
                        log(INFO, f"Inference algorithm: {inference_algorithm.key}")

                        df1 = self._filter_data_for_algorithm(
                            data_df, inference_algorithm
                        )
                        if df1.empty:
                            continue

                        if prediction_type == PredictionType.PREFERENCE_ORDER:
                            self._evaluate_preference_order_results(
                                data_df, df1, inference_algorithm, base_learner_name
                            )
                        else:
                            self._evaluate_binary_vector_results(
                                data_df, df1, inference_algorithm, base_learner_name
                            )

        except Exception as e:
            log(
                ERROR,
                f"Evaluation failed for {dataset_name}: {str(e)}",
                exc_info=True,
            )
            raise

    def _filter_data_for_algorithm(
        self, data_df: pd.DataFrame, inference_algorithm: InferenceAlgorithm
    ) -> pd.DataFrame:
        """Filter data for a specific inference algorithm."""
        return data_df[
            (data_df["preference_order"] == inference_algorithm.order_type.value)
            & (data_df["target_metric"] == inference_algorithm.metric.value)
            & (
                data_df["height"] == inference_algorithm.height
                if inference_algorithm.height is not None
                else data_df["height"].isna()
            )
        ]

    def _evaluate_preference_order_results(
        self,
        data_df: pd.DataFrame,
        df1: pd.DataFrame,
        inference_algorithm: InferenceAlgorithm,
        base_learner_name: str,
    ):
        """Evaluate results for preference order predictions."""
        for order_type in EvaluationConfig.EVALUATION_METRICS[
            PredictionType.PREFERENCE_ORDER
        ].keys():
            if inference_algorithm.order_type.value != order_type.value:
                continue

            for eval_metric in EvaluationConfig.EVALUATION_METRICS[
                PredictionType.PREFERENCE_ORDER
            ][order_type]:
                log(INFO, f"Evaluation metric: {eval_metric}")
                result_folds = self._evaluate_fold(
                    data_df,
                    df1,
                    eval_metric,
                    PredictionType.PREFERENCE_ORDER,
                    order_type,
                )
                res = self.aggregate_results(result_folds)
                self._add_evaluation_result(
                    base_learner_name,
                    inference_algorithm,
                    PredictionType.PREFERENCE_ORDER,
                    eval_metric,
                    res,
                )

    def _evaluate_binary_vector_results(
        self,
        data_df: pd.DataFrame,
        df1: pd.DataFrame,
        inference_algorithm: InferenceAlgorithm,
        base_learner_name: str,
    ):
        """Evaluate results for binary vector predictions."""
        for eval_metric in EvaluationConfig.EVALUATION_METRICS[
            PredictionType.BINARY_VECTOR
        ]:
            log(INFO, f"Evaluation metric: {eval_metric}")
            result_folds = self._evaluate_fold(
                data_df,
                df1,
                eval_metric,
                PredictionType.BINARY_VECTOR,
            )
            res = self.aggregate_results(result_folds)
            self._add_evaluation_result(
                base_learner_name,
                inference_algorithm,
                PredictionType.BINARY_VECTOR,
                eval_metric,
                res,
            )

    def _add_evaluation_result(
        self,
        base_learner_name: str,
        inference_algorithm: InferenceAlgorithm,
        prediction_type: PredictionType,
        eval_metric: EvaluationMetricName,
        res: Dict[str, float],
    ):
        """Add evaluation result to the results list."""
        self.eval_results.append(
            {
                "Base_Learner": base_learner_name,
                "Algorithm": inference_algorithm.key,
                "Algorithm_Metric": inference_algorithm.metric.value,
                "Algorithm_Height": inference_algorithm.height,
                "Algorithm_Order": inference_algorithm.order_type.value,
                "Prediction_Type": prediction_type.value,
                "Metric": eval_metric.value,
                "Mean": res["mean"],
                "Std": res["std"],
            },
        )

    def evaluate_dataset_clr(self, dataset_name: str, noisy_rate: float):
        """Evaluate CLR results for a specific dataset and noise rate."""
        try:
            if self.df is None:
                log(ERROR, "No data loaded. Call load_results() first.")
                return

            log(
                INFO,
                f"Processing results for {dataset_name} with noise rate {noisy_rate}",
            )

            self.eval_results = []

            for base_learner_name in self.df["base_learner_name"].unique():
                log(INFO, f"Processing results for {base_learner_name}")
                data_df = self.get_data_df(dataset_name, base_learner_name)

                for eval_metric in [
                    EvaluationMetricName.HAMMING_ACCURACY,
                    EvaluationMetricName.SUBSET0_1,
                    EvaluationMetricName.F1,
                    EvaluationMetricName.MFRD,
                    EvaluationMetricName.AFRD,
                ]:
                    log(INFO, f"Evaluation metric: {eval_metric}")
                    result_folds = self._evaluate_fold(
                        data_df,
                        data_df,
                        eval_metric,
                        None,
                        is_clr=True,
                    )
                    res = self.aggregate_results(result_folds)
                    self.eval_results.append(
                        {
                            "Base_Learner": base_learner_name,
                            "Algorithm": "CLR",
                            "Metric": eval_metric.value,
                            "Mean": res["mean"],
                            "Std": res["std"],
                        },
                    )

        except Exception as e:
            log(
                ERROR,
                f"Evaluation failed for {dataset_name}: {str(e)}",
                exc_info=True,
            )
            raise

    def evaluate_dataset_br(self, dataset_name: str, noisy_rate: float):
        """Evaluate BR results for a specific dataset and noise rate."""
        try:
            if self.df is None:
                log(ERROR, "No data loaded. Call load_results() first.")
                return

            log(
                INFO,
                f"Processing BR results for {dataset_name} with noise rate {noisy_rate}",
            )

            self.eval_results = []

            for base_learner_name in self.df["base_learner_name"].unique():
                log(INFO, f"Processing results for {base_learner_name}")
                data_df = self.get_data_df(dataset_name, base_learner_name)

                for eval_metric in [
                    EvaluationMetricName.HAMMING_ACCURACY,
                    EvaluationMetricName.SUBSET0_1,
                    EvaluationMetricName.F1,
                    EvaluationMetricName.MFRD,
                    EvaluationMetricName.AFRD,
                ]:
                    log(INFO, f"Evaluation metric: {eval_metric}")
                    result_folds = self._evaluate_fold(
                        data_df,
                        data_df,
                        eval_metric,
                        None,
                        is_br=True,
                    )
                    res = self.aggregate_results(result_folds)
                    self.eval_results.append(
                        {
                            "Base_Learner": base_learner_name,
                            "Algorithm": "BR",
                            "Metric": eval_metric.value,
                            "Mean": res["mean"],
                            "Std": res["std"],
                        },
                    )

        except Exception as e:
            log(
                ERROR,
                f"Evaluation failed for {dataset_name}: {str(e)}",
                exc_info=True,
            )
            raise

    def evaluate_dataset_cc(self, dataset_name: str, noisy_rate: float):
        """Evaluate CC results for a specific dataset and noise rate."""
        try:
            if self.df is None:
                log(ERROR, "No data loaded. Call load_results() first.")
                return

            log(
                INFO,
                f"Processing CC results for {dataset_name} with noise rate {noisy_rate}",
            )

            self.eval_results = []

            for base_learner_name in self.df["base_learner_name"].unique():
                log(INFO, f"Processing results for {base_learner_name}")
                data_df = self.get_data_df(dataset_name, base_learner_name)

                for eval_metric in [
                    EvaluationMetricName.HAMMING_ACCURACY,
                    EvaluationMetricName.SUBSET0_1,
                    EvaluationMetricName.F1,
                    EvaluationMetricName.MFRD,
                    EvaluationMetricName.AFRD,
                ]:
                    log(INFO, f"Evaluation metric: {eval_metric}")
                    result_folds = self._evaluate_fold(
                        data_df,
                        data_df,
                        eval_metric,
                        None,
                        is_cc=True,
                    )
                    res = self.aggregate_results(result_folds)
                    self.eval_results.append(
                        {
                            "Base_Learner": base_learner_name,
                            "Algorithm": "CC",
                            "Metric": eval_metric.value,
                            "Mean": res["mean"],
                            "Std": res["std"],
                        },
                    )

        except Exception as e:
            log(
                ERROR,
                f"Evaluation failed for {dataset_name}: {str(e)}",
                exc_info=True,
            )
            raise

    def save_results(self, output_path: str):
        """Save evaluation results to CSV and Excel."""
        try:
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

    def evaluate_dataset_dataset_level(self, dataset_name: str, noisy_rate: float):
        """Evaluate dataset-level metrics (MEAN_IR and CV_IR) on the entire dataset."""
        try:
            if self.df is None:
                log(ERROR, "No data loaded. Call load_results() first.")
                return

            log(
                INFO,
                f"Processing dataset-level metrics for {dataset_name} with noise rate {noisy_rate}",
            )

            self.eval_results = []

            for base_learner_name in self.df["base_learner_name"].unique():
                log(INFO, f"Processing results for {base_learner_name}")
                data_df = self.get_data_df(dataset_name, base_learner_name)

                try:
                    # Get all test labels from the dataset
                    all_test_labels = np.concatenate(
                        [data_df["Y_test"].values[0] for _ in range(len(data_df))]
                    )

                    # Calculate MEAN_IR
                    mean_ir = self.evaluation_metric.mean_ir(all_test_labels)
                    self.eval_results.append(
                        {
                            "Base_Learner": base_learner_name,
                            "Algorithm": "Dataset",
                            "Metric": EvaluationMetricName.MEAN_IR.value,
                            "Mean": mean_ir,
                            "Std": 0.0,  # No standard deviation for dataset-level metrics
                        },
                    )

                    # Calculate CV_IR
                    cv_ir = self.evaluation_metric.cv_ir(all_test_labels)
                    self.eval_results.append(
                        {
                            "Base_Learner": base_learner_name,
                            "Algorithm": "Dataset",
                            "Metric": EvaluationMetricName.CV_IR.value,
                            "Mean": cv_ir,
                            "Std": 0.0,  # No standard deviation for dataset-level metrics
                        },
                    )

                except Exception as e:
                    log(ERROR, f"Error calculating dataset-level metrics: {str(e)}")
                    continue

        except Exception as e:
            log(
                ERROR,
                f"Dataset-level evaluation failed for {dataset_name}: {str(e)}",
                exc_info=True,
            )
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--results_dir", type=str)
    return parser.parse_args()


def main():
    """Main entry point for evaluation."""
    args = parse_args()

    # Set up results directory
    results_dir = args.results_dir if args.results_dir else "./results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Validate dataset
    if args.dataset is None:
        raise ValueError("Dataset is required")
    dataset_name = ConfigManager.DATASET_KEY[args.dataset]

    log(INFO, f"Args: Dataset: {dataset_name}, Results dir: {results_dir}")

    # Define noise rates
    noisy_rates = [
        0.0,
        # 0.1, 0.2, 0.3
    ]

    # Initialize evaluator
    evaluator = EvaluationFramework(results_dir)

    # Process each noise rate
    for noisy_rate in noisy_rates:
        try:
            # Evaluate dataset-level metrics
            log(INFO, "Start for dataset-level metrics:")
            evaluator.load_results(dataset_name, noisy_rate)
            evaluator.evaluate_dataset_dataset_level(dataset_name, noisy_rate)
            evaluator.save_results(
                f"{results_dir}/evaluation_{dataset_name}_noisy_{noisy_rate}_dataset_level"
            )

            # Evaluate BOPOs
            evaluator.load_results(dataset_name, noisy_rate)
            evaluator.evaluate_dataset(dataset_name, noisy_rate)
            evaluator.save_results(
                f"{results_dir}/evaluation_{dataset_name}_noisy_{noisy_rate}"
            )

            # Evaluate CLR
            log(INFO, "Start for CLR:")
            evaluator.load_results(dataset_name, noisy_rate, is_clr=True)
            evaluator.evaluate_dataset_clr(dataset_name, noisy_rate)
            evaluator.save_results(
                f"{results_dir}/evaluation_{dataset_name}_noisy_{noisy_rate}_clr"
            )

            # Evaluate BR
            log(INFO, "Start for BR:")
            evaluator.load_results(dataset_name, noisy_rate, is_br=True)
            evaluator.evaluate_dataset_br(dataset_name, noisy_rate)
            evaluator.save_results(
                f"{results_dir}/evaluation_{dataset_name}_noisy_{noisy_rate}_br"
            )

            # Evaluate CC
            log(INFO, "Start for CC:")
            evaluator.load_results(dataset_name, noisy_rate, is_cc=True)
            evaluator.evaluate_dataset_cc(dataset_name, noisy_rate)
            evaluator.save_results(
                f"{results_dir}/evaluation_{dataset_name}_noisy_{noisy_rate}_cc"
            )

            log(INFO, f"Evaluation completed successfully for {dataset_name}")

        except Exception as e:
            log(ERROR, f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    main()
