# -predictors- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:54:40 2023

@author: nguyenli_admin
"""

import argparse
import json
import os
import sys
import time

from joblib import Parallel, delayed
from constants import RANDOM_STATE, BaseLearnerName, TargetMetric
from evaluation_metric import EvaluationMetric


from datasets4experiments import Datasets4Experiments
from inference_models import PredictBOPOs, PreferenceOrder
from utils.results_manager import ExperimentResults

# add logging
from logging import basicConfig, INFO, log, ERROR

basicConfig(level=INFO)
# basicConfig(level=ERROR)


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
        # print(predict_results[2])
        indices_vector = predict_results[2]

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
    TOTAL_REPEAT_TIMES: int,
    NUMBER_FOLDS: int,
    base_learners: list[BaseLearnerName],
    dataset_name: str,
    is_clr: bool,
    is_br: bool = False,
    is_cc: bool = False,
):
    results: list[dict] = []
    clr_results: list[dict] = []
    br_results: list[dict] = []
    cc_results: list[dict] = []

    log(
        INFO,
        f"Training for {dataset_name} with {TOTAL_REPEAT_TIMES} repeat times and {NUMBER_FOLDS} folds",
    )

    for base_learner_name in base_learners:
        repeat_times = TOTAL_REPEAT_TIMES
        if base_learner_name == BaseLearnerName.LightGBM:
            repeat_times = 1

        # Run fold for each dataset and each noisy rate and repeat times
        for repeat_time in range(repeat_times):
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
                if not is_clr and not is_br and not is_cc:
                    for order_type in [
                        PreferenceOrder.PRE_ORDER,
                        PreferenceOrder.PARTIAL_ORDER,
                    ]:
                        log(INFO, f"Preference order: {order_type}")
                        train_time1 = time.time()

                        # Initialize the model with the base learner and the preference order
                        predict_BOPOs = PredictBOPOs(
                            base_classifier_name=base_learner_name.value,  # --> Get classifier
                            preference_order=order_type,
                        )

                        # Train the model
                        predict_BOPOs.fit(X_train, Y_train)
                        log(
                            INFO,
                            f"Training time: {(time.time() - train_time1)} seconds",
                        )

                        predict_time1 = time.time()
                        probabilsitic_predictions = predict_BOPOs.predict_proba(
                            X_test, n_labels
                        )

                        log(
                            INFO,
                            f"Prediction time {(time.time() - predict_time1)} seconds",
                        )

                        # Save indices_vector from predict_BOPOs
                        predict_tasks = []
                        for target_metric in [
                            TargetMetric.Hamming,
                            TargetMetric.Subset,
                        ]:
                            for height in [2, None]:
                                predict_tasks.append(
                                    (
                                        probabilsitic_predictions,
                                        n_labels,
                                        n_instances,
                                        target_metric,
                                        height,
                                    )
                                )

                        predict_order_time1 = time.time()

                        predict_results_list = Parallel(n_jobs=-1)(
                            delayed(predict_BOPOs.predict_preference_orders)(*task)
                            for task in predict_tasks
                        )
                        log(
                            INFO,
                            f"Total preference order prediction time {(time.time() - predict_order_time1)} seconds",
                        )

                        save_time1 = time.time()
                        index_result = 0
                        for target_metric in [
                            TargetMetric.Hamming,
                            TargetMetric.Subset,
                        ]:
                            for height in [2, None]:
                                predict_results = predict_results_list[index_result]  # type: ignore
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
                                index_result += 1

                        log(
                            INFO,
                            f"Saving time {(time.time() - save_time1)} seconds",
                        )

                if is_clr:
                    # Support CLR
                    clr_time1 = time.time()
                    clr = PredictBOPOs(
                        base_classifier_name=base_learner_name.value,  # --> Get classifier
                    )
                    clr.fit_CLR(X_train, Y_train)
                    log(
                        INFO,
                        f"CLR Training time: {(time.time() - clr_time1)} seconds",
                    )
                    predict_time1 = time.time()
                    predicted_Y, _ = clr.predict_CLR(X_test, n_labels)
                    log(
                        INFO,
                        f"CLR Prediction time {(time.time() - predict_time1)} seconds",
                    )
                    save_time1 = time.time()
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
                    log(
                        INFO,
                        f"CLR Saving time {(time.time() - save_time1)} seconds",
                    )

                if is_br:
                    # Support Binary Relevance
                    br_time1 = time.time()
                    br = PredictBOPOs(
                        base_classifier_name=base_learner_name.value,
                    )
                    br.fit_BR(X_train, Y_train)
                    log(
                        INFO,
                        f"BR Training time: {(time.time() - br_time1)} seconds",
                    )
                    predict_time1 = time.time()
                    predicted_Y, _ = br.predict_BR(X_test, n_labels)
                    log(
                        INFO,
                        f"BR Prediction time {(time.time() - predict_time1)} seconds",
                    )
                    save_time1 = time.time()
                    update_results(
                        br_results,
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
                    log(
                        INFO,
                        f"BR Saving time {(time.time() - save_time1)} seconds",
                    )

                if is_cc:
                    # Support Classifier Chain
                    cc_time1 = time.time()
                    cc = PredictBOPOs(
                        base_classifier_name=base_learner_name.value,
                    )
                    cc.fit_CC(X_train, Y_train)
                    log(
                        INFO,
                        f"CC Training time: {(time.time() - cc_time1)} seconds",
                    )
                    predict_time1 = time.time()
                    predicted_Y, _ = cc.predict_CC(X_test, n_labels)
                    log(
                        INFO,
                        f"CC Prediction time {(time.time() - predict_time1)} seconds",
                    )
                    save_time1 = time.time()
                    update_results(
                        cc_results,
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
                    log(
                        INFO,
                        f"CC Saving time {(time.time() - save_time1)} seconds",
                    )

    return results, clr_results, br_results, cc_results


def training(
    data_path,
    data_files,
    noisy_rates,
    base_learners,
    TOTAL_REPEAT_TIMES,
    NUMBER_FOLDS,
    results_dir,
):
    load_time1 = time.time()
    experience_dataset = Datasets4Experiments(data_path, data_files)
    experience_dataset.load_datasets()

    log(INFO, f"Loading datasets time taken: {time.time() - load_time1} seconds")
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

            # log(INFO, "Training for Preference Order")

            time1 = time.time()
            res, clr_res, br_res, cc_res = process_dataset(
                experience_dataset,
                dataset_index,
                noisy_rate,
                TOTAL_REPEAT_TIMES,
                NUMBER_FOLDS,
                base_learners,
                dataset_name,
                is_clr=False,
                is_br=False,
                is_cc=False,
            )
            log(
                INFO,
                f"Processing dataset time taken: {time.time() - time1} seconds",
            )
            # log(INFO, "Saving results for Preference Order")

            ExperimentResults.save_results(
                res,
                dataset_name,
                noisy_rate,
                results_dir,
            )

            log(INFO, "Training for CLR")

            _, clr_res, br_res, cc_res = process_dataset(
                experience_dataset,
                dataset_index,
                noisy_rate,
                TOTAL_REPEAT_TIMES,
                NUMBER_FOLDS,
                base_learners,
                dataset_name,
                is_clr=True,
                is_br=False,
                is_cc=False,
            )  # type: ignore

            log(INFO, "Saving results for CLR")
            ExperimentResults.save_results(
                clr_res,
                dataset_name,
                noisy_rate,
                results_dir,
                is_clr=True,
            )

            log(INFO, "Training for BR")
            _, _, br_res, cc_res = process_dataset(
                experience_dataset,
                dataset_index,
                noisy_rate,
                TOTAL_REPEAT_TIMES,
                NUMBER_FOLDS,
                base_learners,
                dataset_name,
                is_clr=False,
                is_br=True,
                is_cc=False,
            )  # type: ignore

            log(INFO, "Saving results for BR")
            ExperimentResults.save_results(
                br_res,
                dataset_name,
                noisy_rate,
                results_dir,
                is_br=True,
            )

            log(INFO, "Training for CC")
            _, _, _, cc_res = process_dataset(
                experience_dataset,
                dataset_index,
                noisy_rate,
                TOTAL_REPEAT_TIMES,
                NUMBER_FOLDS,
                base_learners,
                dataset_name,
                is_clr=False,
                is_br=False,
                is_cc=True,
            )  # type: ignore

            log(INFO, "Saving results for CC")
            ExperimentResults.save_results(
                cc_res,
                dataset_name,
                noisy_rate,
                results_dir,
                is_cc=True,
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


def run_training():
    data_path = "./data/"
    results_dir = "./results"

    arg = argparse.ArgumentParser()
    arg.add_argument("--dataset", type=str)
    arg.add_argument("--results_dir", type=str)
    args = arg.parse_args()

    if args.results_dir is None:
        results_dir = "./results"
    else:
        results_dir = args.results_dir

    if args.dataset.lower() == "chd_49":
        data_files = [
            {
                "dataset_name": "CHD_49.arff",
                "n_labels_set": 6,
            }
        ]
    elif args.dataset.lower() == "emotions":
        data_files = [
            {
                "dataset_name": "emotions.arff",
                "n_labels_set": 6,
            }
        ]
    elif args.dataset.lower() == "scene":
        data_files = [
            {
                "dataset_name": "scene.arff",
                "n_labels_set": 6,
            }
        ]
    elif args.dataset.lower() == "VirusPseAAC".lower():
        data_files = [
            {
                "dataset_name": "VirusPseAAC.arff",
                "n_labels_set": 6,
            }
        ]
    # elif args.dataset.lower() == "yelp":
    #     data_files = [
    #         {
    #             "dataset_name": "Yelp.arff", # has nominal attributes
    #             "n_labels_set": 8,
    #         }
    #     ]
    # elif args.dataset.lower() == "birds":
    #     data_files = [
    #         {
    #             "dataset_name": "birds.arff", # has nominal attributes
    #             "n_labels_set": 19,
    #         }
    #     ]
    elif args.dataset.lower() == "yeast":
        data_files = [
            {
                "dataset_name": "Yeast.arff",
                "n_labels_set": 14,
            }
        ]
    elif args.dataset.lower() == "water-quality":
        data_files = [
            {
                "dataset_name": "Water-quality.arff",
                "n_labels_set": 14,
            }
        ]
    elif args.dataset.lower() == "HumanPseAAC".lower():
        data_files = [
            {
                "dataset_name": "HumanPseAAC.arff",
                "n_labels_set": 14,
            }
        ]
    elif args.dataset.lower() == "GpositivePseAAC".lower():
        data_files = [
            {
                "dataset_name": "GpositivePseAAC.arff",
                "n_labels_set": 4,
            }
        ]
    elif args.dataset.lower() == "PlantPseAAC".lower():
        data_files = [
            {
                "dataset_name": "PlantPseAAC.arff",
                "n_labels_set": 12,
            }
        ]
    else:
        raise ValueError(f"Dataset {args.dataset} not found")

    noisy_rates = [
        # 0.0,
        0.1,
        # 0.2,
        # 0.3,
    ]
    base_learners = [
        BaseLearnerName.LightGBM,
        # BaseLearnerName.RF,
        # BaseLearnerName.XGBoost,
        # BaseLearnerName.ETC,
    ]

    TOTAL_REPEAT_TIMES = 1
    NUMBER_FOLDS = 2

    time1 = time.time()

    training(
        data_path,
        data_files,
        noisy_rates,
        base_learners,
        TOTAL_REPEAT_TIMES,
        NUMBER_FOLDS,
        results_dir,
    )

    time2 = time.time()
    log(INFO, f"Total time taken: {time2 - time1} seconds")


# for a quick test
if __name__ == "__main__":
    run_training()
