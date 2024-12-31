# -predictors- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:54:40 2023

@author: nguyenli_admin
"""

import json
from constants import RANDOM_STATE, BaseLearnerName, TargetMetric
from estimator import Estimator
from evaluation_metric import EvaluationMetric, EvaluationMetricName


from datasets4experiments import Datasets4Experiments
from inference_models import PredictBOPOs, PreferenceOrder

# add logging
from logging import basicConfig, INFO, log

basicConfig(level=INFO)


def update_results(
    Y_test,
    results,
    dataset_index,
    repeat_time,
    fold,
    noisy_rate,
    base_learner_name,
    predict_results,
    target_metric,
    order_type,
):
    if dataset_index not in results:
        results[dataset_index] = {}
    if f"{noisy_rate}" not in results[dataset_index]:
        results[dataset_index][f"{noisy_rate}"] = {}
    if base_learner_name not in results[dataset_index][f"{noisy_rate}"]:
        results[dataset_index][f"{noisy_rate}"][base_learner_name] = {}
    if (
        f"{repeat_time}_{fold}"
        not in results[dataset_index][f"{noisy_rate}"][base_learner_name]
    ):
        results[dataset_index][f"{noisy_rate}"][base_learner_name][
            f"{repeat_time}_{fold}"
        ] = {}

    results[dataset_index][f"{noisy_rate}"][base_learner_name][
        f"{repeat_time}_{fold}"
    ] = {
        "Y_test": list(Y_test),
        "predict_results": list(predict_results),
        "target_metric": target_metric,
        "preference_order": order_type,
    }

    return results


def process_dataset(
    experiment_dataset: Datasets4Experiments,
    dataset_index: int,
    noisy_rate: float,
    repeat_time: int,
    NUMBER_FOLDS: int,
    base_learners: list[BaseLearnerName],
):
    results = {
        # [dataset_index][noisy_rate][base_learner]: mean and std of evaluation metrics
    }

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
                    # PreferenceOrder.PARTIAL_ORDER,
                ]:
                    # TODO: how to handle other preference orders?
                    log(INFO, f"Preference order: {order_type}")

                    # Initialize the model with the base learner and the preference order
                    predict_BOPOs = PredictBOPOs(
                        base_classifier_name=base_learner_name.value,  # --> Get classifier
                        preference_order=order_type,
                    )

                    # Train the model
                    predict_BOPOs.fit(X_train, Y_train)

                    log(INFO, f"PredictBOPOs: {predict_BOPOs}")

                    # Linh: For shared configurations, i.e., experiments with the same type
                    # of preference orders, which can be either partial or preorders, we can
                    # put an option to call this function for the first configuration, e.g.,
                    # with Hamming accuracy
                    # For the next configurations, we re-use the pre-trained models.

                    # Predict the test set
                    n_instances = X_test.shape[0]
                    n_labels = Y_test.shape[1]

                    probabilsitic_predictions = predict_BOPOs.predict_proba(
                        X_test, n_labels
                    )

                    for target_metric in [TargetMetric.Hamming, TargetMetric.Subset]:
                        predict_results = predict_BOPOs.predict_preference_orders(
                            probabilsitic_predictions,
                            n_labels,
                            n_instances,
                            target_metric,
                        )

                        results = update_results(
                            Y_test,
                            results,
                            dataset_index,
                            repeat_time,
                            fold,
                            noisy_rate,
                            base_learner_name.value,
                            predict_results,
                            target_metric.value,
                            order_type.value,
                        )

    return results


def main(
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
            results = process_dataset(
                experience_dataset,
                dataset_index,
                noisy_rate,
                TOTAL_REPEAT_TIMES,
                NUMBER_FOLDS,
                base_learners,
            )

            log(INFO, f"Results: {results}")
            # Save to file
            with open(f"results_1.txt", "w") as f:
                f.write(str(results))

            # For each evaluation metric
            for metric in EvaluationMetricName:
                log(INFO, f"Evaluation metric: {metric}")
                # Do evaluation here
                # TODO: Do this evaluation


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

    main(
        data_path,
        data_files,
        n_labels_set,
        noisy_rates,
        base_learners,
        TOTAL_REPEAT_TIMES,
        NUMBER_FOLDS,
    )
