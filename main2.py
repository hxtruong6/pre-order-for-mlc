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
    results,
    Y_test,
    predict_results,
    repeat_time,
    fold,
    base_learner_name,
    target_metric,
    order_type,
    height,
    dataset_name,
    noisy_rate,
):

    indices_vector = None
    if len(predict_results) == 3:
        indices_vector = list(predict_results[2])

    data = {
        "Y_test": Y_test.tolist(),
        "Y_predicted": list(predict_results[0]),
        "Y_BOPOs": list(predict_results[1]),
        "indices_vector": indices_vector,
        "target_metric": target_metric,
        "preference_order": order_type,
        "height": height,
        "repeat_time": repeat_time,
        "fold": fold,
        "dataset_name": dataset_name,
        "base_learner_name": base_learner_name,
        "noisy_rate": noisy_rate,
    }

    results.append(data)


def process_dataset(
    experiment_dataset: Datasets4Experiments,
    dataset_index: int,
    noisy_rate: float,
    repeat_time: int,
    NUMBER_FOLDS: int,
    base_learners: list[BaseLearnerName],
    dataset_name: str,
):
    results: list[dict] = []
    clr_results: list[dict] = []

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
                n_instances = X_test.shape[0]
                n_labels = Y_test.shape[1]
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

                    # log(INFO, f"PredictBOPOs: {predict_BOPOs}")

                    probabilsitic_predictions = predict_BOPOs.predict_proba(
                        X_test, n_labels
                    )

                    # Save indices_vector from predict_BOPOs

                    for target_metric in [TargetMetric.Hamming, TargetMetric.Subset]:
                        for height in [2, None]:
                            predict_results = predict_BOPOs.predict_preference_orders(
                                probabilsitic_predictions,
                                n_labels,
                                n_instances,
                                target_metric,
                                height,
                            )

                            update_results(
                                results,
                                Y_test,
                                predict_results,
                                repeat_time,
                                fold,
                                base_learner_name.value,
                                target_metric.value,
                                order_type.value,
                                height,
                                dataset_name,
                                noisy_rate,
                            )

                # Support CLR
                clr = PredictBOPOs(
                    base_classifier_name=base_learner_name.value,  # --> Get classifier
                )
                clr.fit_CLR(X_train, Y_train)
                predicted_Y, _ = clr.predict_CLR(X_test, n_labels)
                update_results(
                    clr_results,
                    Y_test,
                    [predicted_Y, []],
                    repeat_time,
                    fold,
                    base_learner_name.value,
                    None,
                    None,
                    None,
                    dataset_name,
                    noisy_rate,
                )

    return results, clr_results


def training(
    data_path,
    data_files,
    n_labels_set,
    noisy_rates,
    base_learners,
    TOTAL_REPEAT_TIMES,
    NUMBER_FOLDS,
):
    experience_dataset = Datasets4Experiments(data_path, data_files, n_labels_set)
    experience_dataset.load_datasets()

    # Run for each dataset
    for dataset_index in range(experience_dataset.get_length()):
        dataset_name = experience_dataset.get_dataset_name(dataset_index)
        log(
            INFO,
            f"Dataset: {dataset_index}: {dataset_name}",
        )
        # Run for each noisy rate
        for noisy_rate in noisy_rates:
            log(INFO, f"Noisy rate: {noisy_rate}")
            res, clr_res = process_dataset(
                experience_dataset,
                dataset_index,
                noisy_rate,
                TOTAL_REPEAT_TIMES,
                NUMBER_FOLDS,
                base_learners,
                dataset_name,
            )

            log(INFO, f"Result length: {len(res)}")

            ExperimentResults.save_results(
                res,
                dataset_name,
                noisy_rate,
            )

            ExperimentResults.save_results(
                clr_res,
                dataset_name,
                noisy_rate,
                is_clr=True,
            )


# def evaluate_dataset(dataset_name, noisy_rate, results):
#     inference_algorithms = {
#         "IA1": "Preorders + Hamming + Height = None",
#         "IA2": "Preorders + Hamming + Height = 2",
#         "IA3": "Preorders + Subset + Height = None",
#         "IA4": "Preorders + Subset + Height = 2",
#         "IA5": "Partial_orders + Hamming + Height = None",
#         "IA6": "Partial_orders + Hamming + Height = 2",
#         "IA7": "Partial_orders + Subset + Height = None",
#         "IA8": "Partial_orders + Subset + Height = 2",
#     }

#     prediction_types = {
#         "PT1": "Preference order",
#         "PT2": "Binary vector",
#     }

#     evaluation_metrics = {
#         "EM1": "Hamming",
#         "EM2": "Subset",
#         "EM3": "F measure",
#     }

#     for inference_algorithm in inference_algorithms:
#         for prediction_type in prediction_types:
#             if prediction_type == "PT1":
#                 for evaluation_metric in evaluation_metrics:
#                     log(
#                         INFO,
#                         f"Inference algorithm: {inference_algorithm}, Prediction type: {prediction_type}, Evaluation metric: {evaluation_metric}",
#                     )
#             elif inference_algorithm in ["IA1", "IA2", "IA3", "IA4"]:
#                 for evaluation_metric in [TargetMetric.Hamming, TargetMetric.Subset]:
#                     pass
#             elif inference_algorithm in ["IA5", "IA6", "IA7", "IA8"]:
#                 for evaluation_metric in [TargetMetric.Hamming, TargetMetric.Subset]:
#                     pass


# def evaluating(saved_path):
#     with open(saved_path, "r") as f:
#         results = json.load(f)
#     """
#     TODO: Create a dictionary of possible configuration (8 inference algorithms, 2 prediction types, 7 evaluation metrics)
#     8 inference algorithms:
#        - IA1: Preorders + Hamming + Height = None
#        - IA2: Preorders + Hamming + Height = 2
#        - IA3: Preorders + Subset + Height = None
#        - IA4: Preorders + Subset + Height = 2

#        - IA5: Partial_orders + Hamming + Height = None
#        - IA6: Partial_orders + Hamming + Height = 2
#        - IA7: Partial_orders + Subset + Height = None
#        - IA8: Partial_orders + Subset + Height = 2

#     2 prediction types per inference algorithms:
#        - PT1: Preference order
#        - PT2: Binary vector

#     7 evaluation metrics:
#        - PT2: Hamming or Subset or F measure: hamming_accuracy, subset0_1, f1
#        - PT1:
#             - IA1 - IA4: Hamming or Subset for preorder: hamming_accuracy_PRE_ORDER, subset0_1_accuracy_PRE_ORDER
#             - IA5 - IA8: Hamming or Subset for partial_order: hamming_accuracy_PARTIAL_ORDER, subset0_1_accuracy_PARTIAL_ORDER

#     We have table:
#         - 1 dataset X number of BASE_LEARNER X number of NOISY_RATE:
#             - PT2: 1 table (8 inference algorithms * 3 evaluation metrics ( hamming_accuracy, subset0_1, f1))
#             - PT1: 2 table:
#                 - PRE_ORDER: 4 inference algorithms * 2 evaluation metrics ( hamming_accuracy_PRE_ORDER, subset0_1_accuracy_PRE_ORDER)
#                 - PARTIAL_ORDER: 4 inference algorithms * 2 evaluation metrics ( hamming_accuracy_PARTIAL_ORDER, subset0_1_accuracy_PARTIAL_ORDER)

#     Fold: np.mean(), np.std()
#     """

# for a quick test
if __name__ == "__main__":
    # Configuration
    data_path = "./data/"
    data_files = [
        # "emotions.arff",
        "CHD_49.arff",
        # "scene.arff",
        # "Yeast.arff",
        # "Water-quality.arff",
    ]
    n_labels_set = [6, 6, 6, 14, 14]  # number of labels in each dataset
    noisy_rates = [0.0, 0.1, 0.2, 0.3]
    base_learners = [
        BaseLearnerName.RF,
        BaseLearnerName.XGBoost,
        BaseLearnerName.ETC,
        BaseLearnerName.LightGBM,
    ]

    TOTAL_REPEAT_TIMES = 5
    NUMBER_FOLDS = 5

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
