import csv
import json
from logging import ERROR, basicConfig, INFO, log

import os
import pickle
import numpy as np
from collections import defaultdict


basicConfig(level=INFO)

# Define inference algorithms
INFERENCE_ALGORITHMS = [
    {"type": "Pre_orders", "metric": "Hamming", "height": None},
    {"type": "Pre_orders", "metric": "Hamming", "height": 2},
    {"type": "Pre_orders", "metric": "Subset", "height": None},
    {"type": "Pre_orders", "metric": "Subset", "height": 2},
    {"type": "Partial_orders", "metric": "Hamming", "height": None},
    {"type": "Partial_orders", "metric": "Hamming", "height": 2},
    {"type": "Partial_orders", "metric": "Subset", "height": None},
    {"type": "Partial_orders", "metric": "Subset", "height": 2},
]

# Define prediction types
PREDICTION_TYPES = ["Preference_order", "Binary_vector"]

# Define evaluation metrics
EVALUATION_METRICS = {
    "Binary_vector": ["hamming_accuracy", "subset0_1", "f1"],
    "Preference_order": {
        "Pre_orders": ["hamming_accuracy_PRE_ORDER", "subset0_1_accuracy_PRE_ORDER"],
        "Partial_orders": [
            "hamming_accuracy_PARTIAL_ORDER",
            "subset0_1_accuracy_PARTIAL_ORDER",
        ],
    },
}


class EvaluationFramework:
    """
    Framework for evaluating datasets with different inference algorithms,
    prediction types, and noisy rates.
    """

    def __init__(self, dataset, base_learners, noisy_rates):
        self.dataset = dataset
        self.base_learners = base_learners
        self.noisy_rates = noisy_rates
        self.results = defaultdict(
            lambda: defaultdict(list)
        )  # Store results in a nested dictionary

    def evaluate(self, inference_algo, prediction_type, metrics, fold_data):
        """
        Evaluate given fold data with specified metrics.
        """
        metric_results = {}
        for metric in metrics:
            func = getattr(self, f"_{metric}", None)
            log(INFO, f"evaluate func: {func}")
            metric_results[metric] = np.random.random()
            if func:
                metric_results[metric] = func(fold_data)
            else:
                log(WARNING, f"Metric {metric} not found")
        return metric_results

    def _hamming_accuracy(self, fold_data):
        return np.random.random()

    def _subset_accuracy(self, fold_data):
        return np.random.random()

    def _f1(self, fold_data):
        return np.random.random()

    def run(self):
        """
        Run evaluations for all configurations.
        """
        for algo in INFERENCE_ALGORITHMS:
            algo_type = algo["type"]
            metrics = EVALUATION_METRICS.get("Preference_order", {}).get(
                algo_type, EVALUATION_METRICS["Binary_vector"]
            )
            for pred_type in PREDICTION_TYPES:
                for base_learner in self.base_learners:
                    for noisy_rate in self.noisy_rates:
                        fold_results = [
                            self.evaluate(
                                algo, pred_type, metrics, self._generate_mock_data()
                            )
                            for _ in range(5)  # Simulating 5 folds
                        ]
                        print(fold_results)
                        self._aggregate_results(
                            algo, pred_type, base_learner, noisy_rate, fold_results
                        )

    def _aggregate_results(
        self, algo, pred_type, base_learner, noisy_rate, fold_results
    ):
        """
        Aggregate fold results (mean and std).
        """
        aggregated_metrics = {}
        for metric in fold_results[0].keys():
            metric_values = [fold[metric] for fold in fold_results]
            print(metric_values)
            aggregated_metrics[f"{metric}_mean"] = np.mean(metric_values)
            aggregated_metrics[f"{metric}_std"] = np.std(metric_values)
        # print(aggregated_metrics)
        # Store results
        key = f"{algo['type']}_{algo['metric']}_{algo['height']}"
        print(key)
        self.results[key][pred_type].append(
            {
                "base_learner": base_learner,
                "noisy_rate": noisy_rate,
                **aggregated_metrics,
            }
        )

    def _generate_mock_data(self):
        """
        Generate mock data for folds (replace with actual data loading logic).
        """
        return np.random.random(size=10)  # Example data

    def save_to_csv(self, csv_file):
        """
        Save results to a CSV file.
        """
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)

            # Write the header
            header = [
                "Algorithm Type",
                "Metric",
                "Height",
                "Prediction Type",
                "Base Learner",
                "Noisy Rate",
                "Metric",
                "Mean",
                "Std",
            ]
            writer.writerow(header)

            # Write the data rows
            for algo_key, pred_types in self.results.items():
                algo_parts = algo_key.split("_")
                algo_type, metric, height = algo_parts[0], algo_parts[1], algo_parts[2]
                for pred_type, evaluations in pred_types.items():
                    for eval_entry in evaluations:
                        base_learner = eval_entry["base_learner"]
                        noisy_rate = eval_entry["noisy_rate"]
                        for metric_name, value in eval_entry.items():
                            if metric_name.endswith("_mean"):
                                metric_base_name = metric_name.replace("_mean", "")
                                mean = value
                                std = eval_entry.get(f"{metric_base_name}_std", 0.0)
                                writer.writerow(
                                    [
                                        algo_type,
                                        metric,
                                        height,
                                        pred_type,
                                        base_learner,
                                        noisy_rate,
                                        metric_base_name,
                                        mean,
                                        std,
                                    ]
                                )


# Running the evaluation
if __name__ == "__main__":
    print("Hello")
    # dataset = "sample_dataset"
    # base_learners = ["Learner1", "Learner2"]
    # noisy_rates = [0.0, 0.1]
    #  Read prediction results from pickle file
    path = "results/new/dataset_emotions__noisy_0.0.pkl"
    if not os.path.exists(path):
        log(ERROR, f"File not found: {path}")
        exit(1)

    try:
        with open(path, "rb") as file:
            results = pickle.load(file)

        for i, base_learner in enumerate(results):

            print("base: ", base_learner)
            for repeat_time in results[base_learner].keys():
                print("repeat_time: ", repeat_time)
                repeat, fold = repeat_time.split("__")
                print("repeat: ", repeat)
                print("fold: ", fold)
                for algo in results[base_learner][repeat_time].keys():
                    metric, order_type, height = algo.split("__")
                    print("metric: ", metric)
                    print("order_type: ", order_type)
                    print("height: ", height)
                    # for fold in results[base_learner][repeat_time].keys():
                    #     print("fold: ", fold)
                    #     # for algo in results[base_learner][repeat_time][fold].keys():
                    #     #     print("algo: ", algo)

            break

    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error loading pickle file: {e}")

    # return 0

    # framework = EvaluationFramework(dataset, base_learners, noisy_rates)
    # framework.run()

    # # Save results to JSON
    # with open("evaluation_results.json", "w") as f:
    #     json.dump(framework.results, f, indent=4)

    # # Save results to a CSV file
    # output_csv = "evaluation_results.csv"
    # framework.save_to_csv(output_csv)
    # print(f"Results saved to {output_csv}")
