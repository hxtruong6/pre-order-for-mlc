# -predictors- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:54:40 2023

@author: nguyenli_admin
"""

import json
from constants import RANDOM_STATE, BaseLearnerName, TargetMetric
from evaluation_metric import EvaluationMetric


from datasets4experiments import Datasets4Experiments
from inference_models import PredictBOPOs, PreferenceOrder
from utils.results_manager import ExperimentResults

# add logging
from logging import basicConfig, INFO, log

basicConfig(level=INFO)


def update_results(
    Y_test,
    results,
    repeat_time,
    fold,
    base_learner_name,
    predict_results,
    target_metric,
    order_type,
    height,
):
    log(
        INFO,
        f"Target metric: {target_metric}, Order type: {order_type}, Height: {height}",
    )

    repeat_fold = f"repeat_{repeat_time}__fold_{fold}"

    metric_preferenceOrder_height = (
        f"{target_metric}__{order_type}__height_{height if height else 'None'}"
    )

    if base_learner_name not in results:
        results[base_learner_name] = {}
    if repeat_fold not in results[base_learner_name]:
        results[base_learner_name][repeat_fold] = {}

    if metric_preferenceOrder_height not in results[base_learner_name][repeat_fold]:
        results[base_learner_name][repeat_fold][metric_preferenceOrder_height] = []

    data = {
        "Y_test": Y_test,
        "predict_results": predict_results,
        "target_metric": target_metric,
        "preference_order": order_type,
        "height": height,
        "repeat_time": repeat_time,
        "fold": fold,
    }

    results[base_learner_name][repeat_fold][metric_preferenceOrder_height].append(data)

    return results


def process_dataset(
    experiment_dataset: Datasets4Experiments,
    dataset_index: int,
    noisy_rate: float,
    repeat_time: int,
    NUMBER_FOLDS: int,
    base_learners: list[BaseLearnerName],
):
    results = {}

    for base_learner_name in base_learners:

        # Run fold for each dataset and each noisy rate and repeat times
        for repeat_time in range(TOTAL_REPEAT_TIMES):
            log(INFO, f"Dataset: {dataset_index}, Repeat time: {repeat_time}")
            for fold, (X_train, Y_train, X_test, Y_test) in enumerate(
                experiment_dataset.kfold_split_with_noise(
                    dataset_index=dataset_index,
                    n_splits=NUMBER_FOLDS,
                    noisy_rate=noisy_rate,
                    random_state=RANDOM_STATE,
                )
            ):
                log(
                    INFO,
                    f"Fold: {fold}/{NUMBER_FOLDS} - {noisy_rate}",
                )
                for order_type in [
                    PreferenceOrder.PRE_ORDER,
                    PreferenceOrder.PARTIAL_ORDER,
                ]:
                    log(INFO, f"Preference order: {order_type}")

                    # Initialize the model with the base learner and the preference order
                    predict_BOPOs = PredictBOPOs(
                        base_classifier_name=base_learner_name.value,  # --> Get classifier
                        preference_order=order_type,
                    )

                    # Train the model
                    predict_BOPOs.fit(X_train, Y_train)

                    log(INFO, f"PredictBOPOs: {predict_BOPOs}")

                    n_instances = X_test.shape[0]
                    n_labels = Y_test.shape[1]

                    probabilsitic_predictions = predict_BOPOs.predict_proba(
                        X_test, n_labels
                    )

                    for target_metric in [TargetMetric.Hamming, TargetMetric.Subset]:
                        for height in [2, None]:
                            predict_results = predict_BOPOs.predict_preference_orders(
                                probabilsitic_predictions,
                                n_labels,
                                n_instances,
                                target_metric,
                                height,
                            )

                            results = update_results(
                                Y_test,
                                results,
                                repeat_time,
                                fold,
                                base_learner_name.value,
                                predict_results,
                                target_metric.value,
                                order_type.value,
                                height,
                            )
    return results


def training(
    data_path,
    data_files,
    n_labels_set,
    noisy_rates,
    base_learners,
    TOTAL_REPEAT_TIMES,
    NUMBER_FOLDS,
):
    eval_metric = EvaluationMetric()

    experience_dataset = Datasets4Experiments(data_path, data_files, n_labels_set)
    experience_dataset.load_datasets()

    # Run for each dataset
    for dataset_index in range(experience_dataset.get_length()):
        log(
            INFO,
            f"Dataset: {dataset_index}: {experience_dataset.get_dataset_name(dataset_index)}",
        )
        # Run for each noisy rate
        for noisy_rate in noisy_rates:
            log(INFO, f"Noisy rate: {noisy_rate}")
            res = process_dataset(
                experience_dataset,
                dataset_index,
                noisy_rate,
                TOTAL_REPEAT_TIMES,
                NUMBER_FOLDS,
                base_learners,
            )

            ExperimentResults.save_results(
                res,
                experience_dataset.get_dataset_name(dataset_index),
                noisy_rate,
            )


def evaluate_dataset(dataset_name, noisy_rate, results):
    inference_algorithms = {
        "IA1": "Preorders + Hamming + Height = None",
        "IA2": "Preorders + Hamming + Height = 2",
        "IA3": "Preorders + Subset + Height = None",
        "IA4": "Preorders + Subset + Height = 2",
        "IA5": "Partial_orders + Hamming + Height = None",
        "IA6": "Partial_orders + Hamming + Height = 2",
        "IA7": "Partial_orders + Subset + Height = None",
        "IA8": "Partial_orders + Subset + Height = 2",
    }

    prediction_types = {
        "PT1": "Preference order",
        "PT2": "Binary vector",
    }

    evaluation_metrics = {
        "EM1": "Hamming",
        "EM2": "Subset",
        "EM3": "F measure",
    }

    for inference_algorithm in inference_algorithms:
        for prediction_type in prediction_types:
            if prediction_type == "PT1":
                for evaluation_metric in evaluation_metrics:
                    log(
                        INFO,
                        f"Inference algorithm: {inference_algorithm}, Prediction type: {prediction_type}, Evaluation metric: {evaluation_metric}",
                    )
            elif inference_algorithm in ["IA1", "IA2", "IA3", "IA4"]:
                for evaluation_metric in [TargetMetric.Hamming, TargetMetric.Subset]:
                    pass
            elif inference_algorithm in ["IA5", "IA6", "IA7", "IA8"]:
                for evaluation_metric in [TargetMetric.Hamming, TargetMetric.Subset]:
                    pass


def evaluating(saved_path):
    with open(saved_path, "r") as f:
        results = json.load(f)
    """
    TODO: Create a dictionary of possible configuration (8 inference algorithms, 2 prediction types, 7 evaluation metrics)
    8 inference algorithms:
       - IA1: Preorders + Hamming + Height = None
       - IA2: Preorders + Hamming + Height = 2
       - IA3: Preorders + Subset + Height = None
       - IA4: Preorders + Subset + Height = 2

       - IA5: Partial_orders + Hamming + Height = None
       - IA6: Partial_orders + Hamming + Height = 2
       - IA7: Partial_orders + Subset + Height = None
       - IA8: Partial_orders + Subset + Height = 2

    2 prediction types per inference algorithms:
       - PT1: Preference order
       - PT2: Binary vector

    7 evaluation metrics:
       - PT2: Hamming or Subset or F measure: hamming_accuracy, subset0_1, f1
       - PT1:
            - IA1 - IA4: Hamming or Subset for preorder: hamming_accuracy_PRE_ORDER, subset0_1_accuracy_PRE_ORDER
            - IA5 - IA8: Hamming or Subset for partial_order: hamming_accuracy_PARTIAL_ORDER, subset0_1_accuracy_PARTIAL_ORDER
    
    We have table:
        - 1 dataset X number of BASE_LEARNER X number of NOISY_RATE:
            - PT2: 1 table (8 inference algorithms * 3 evaluation metrics ( hamming_accuracy, subset0_1, f1))
            - PT1: 2 table:
                - PRE_ORDER: 4 inference algorithms * 2 evaluation metrics ( hamming_accuracy_PRE_ORDER, subset0_1_accuracy_PRE_ORDER)
                - PARTIAL_ORDER: 4 inference algorithms * 2 evaluation metrics ( hamming_accuracy_PARTIAL_ORDER, subset0_1_accuracy_PARTIAL_ORDER)

    Fold: np.mean(), np.std()
    """

# for a quick test
if __name__ == "__main__":
    # Configuration
    data_path = "./data/"
    data_files = [
        "emotions.arff",
        # "CHD_49.arff",
        # "scene.arff",
        # "Yeast.arff",
        # "Water-quality.arff",
    ]
    n_labels_set = [6, 6, 6, 14, 14]  # number of labels in each dataset
    noisy_rates = [
        0.0,
        # 0.2,
        # 0.4,
    ]
    base_learners = [BaseLearnerName.RF]

    TOTAL_REPEAT_TIMES = 1
    NUMBER_FOLDS = 2

    training(
        data_path,
        data_files,
        n_labels_set,
        noisy_rates,
        base_learners,
        TOTAL_REPEAT_TIMES,
        NUMBER_FOLDS,
    )

    # saved_path = "./results/results_1_2.txt"
    # evaluating(saved_path)
