import time
from typing import List, Dict, Tuple
from logging import INFO, ERROR, log
from joblib import Parallel, delayed
from config import TrainingConfig, DatasetConfig, AlgorithmType
from datasets4experiments import Datasets4Experiments
from inference_models import PredictBOPOs, PreferenceOrder
from utils.results_manager import ExperimentResults
from constants import BaseLearnerName, RANDOM_STATE, TargetMetric


class TrainingOrchestrator:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.experiment_dataset: Datasets4Experiments
        self.results = {}

    def setup(self, dataset_config: DatasetConfig):
        """Initialize datasets and other resources."""
        log(INFO, "Loading datasets...")
        load_time = time.time()
        self.experiment_dataset = Datasets4Experiments(
            self.config.data_path,
            [
                {
                    "dataset_name": dataset_config.file,
                    "n_labels_set": dataset_config.n_labels,
                }
            ],
        )
        self.experiment_dataset.load_datasets()
        log(INFO, f"Loading datasets time taken: {time.time() - load_time} seconds")

    def process_dataset(
        self,
        dataset_index: int,
        noisy_rate: float,
        dataset_name: str,
        algorithm: AlgorithmType,
    ) -> List[Dict]:
        """Process a single dataset with given noise rate for a specific algorithm."""
        results = []

        log(
            INFO,
            f"Training {algorithm.value} for {dataset_name} with {self.config.total_repeat_times} repeat times and {self.config.number_folds} folds",
        )

        for base_learner_name in self.config.base_learners:
            repeat_times = self.config.total_repeat_times
            if base_learner_name == BaseLearnerName.LightGBM:
                repeat_times = 1

            for repeat_time in range(repeat_times):
                log(INFO, f"Dataset: {dataset_index}, Repeat time: {repeat_time}")

                for fold, (X_train, Y_train, X_test, Y_test) in enumerate(
                    self.experiment_dataset.kfold_split_with_noise(
                        dataset_index=dataset_index,
                        n_splits=self.config.number_folds,
                        noisy_rate=noisy_rate,
                        random_state=RANDOM_STATE,
                    )
                ):
                    n_instances = X_test.shape[0]
                    n_labels = Y_test.shape[1]
                    log(
                        INFO,
                        f"Fold: {fold+1}/{self.config.number_folds} - {noisy_rate}",
                    )

                    if algorithm == AlgorithmType.BOPOS:
                        results.extend(
                            self._process_bopos(
                                X_train,
                                Y_train,
                                X_test,
                                Y_test,
                                base_learner_name,
                                repeat_time,
                                fold,
                                dataset_name,
                                noisy_rate,
                            )
                        )
                    elif algorithm == AlgorithmType.CLR:
                        results.extend(
                            self._process_clr(
                                X_train,
                                Y_train,
                                X_test,
                                Y_test,
                                base_learner_name,
                                repeat_time,
                                fold,
                                dataset_name,
                                noisy_rate,
                            )
                        )
                    elif algorithm == AlgorithmType.BR:
                        results.extend(
                            self._process_br(
                                X_train,
                                Y_train,
                                X_test,
                                Y_test,
                                base_learner_name,
                                repeat_time,
                                fold,
                                dataset_name,
                                noisy_rate,
                            )
                        )
                    elif algorithm == AlgorithmType.CC:
                        results.extend(
                            self._process_cc(
                                X_train,
                                Y_train,
                                X_test,
                                Y_test,
                                base_learner_name,
                                repeat_time,
                                fold,
                                dataset_name,
                                noisy_rate,
                            )
                        )

        return results

    def _process_bopos(
        self,
        X_train,
        Y_train,
        X_test,
        Y_test,
        base_learner_name,
        repeat_time,
        fold,
        dataset_name,
        noisy_rate,
    ):
        """Process BOPOs training and prediction."""
        results = []
        for order_type in [PreferenceOrder.PRE_ORDER, PreferenceOrder.PARTIAL_ORDER]:
            log(INFO, f"Preference order: {order_type}")
            train_time1 = time.time()

            predict_BOPOs = PredictBOPOs(
                base_classifier_name=base_learner_name.value,
                preference_order=order_type,
            )
            predict_BOPOs.fit(X_train, Y_train)
            log(INFO, f"Training time: {(time.time() - train_time1)} seconds")

            predict_time1 = time.time()
            probabilistic_predictions = predict_BOPOs.predict_proba(
                X_test, Y_test.shape[1]
            )
            log(INFO, f"Prediction time {(time.time() - predict_time1)} seconds")

            predict_tasks = []
            for target_metric in [TargetMetric.Hamming, TargetMetric.Subset]:
                for height in [2, None]:
                    predict_tasks.append(
                        (
                            probabilistic_predictions,
                            Y_test.shape[1],
                            X_test.shape[0],
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
            for target_metric in [TargetMetric.Hamming, TargetMetric.Subset]:
                for height in [2, None]:
                    predict_results = predict_results_list[index_result]  # type: ignore
                    self._update_results(
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
            log(INFO, f"Saving time {(time.time() - save_time1)} seconds")

        return results

    def _process_clr(
        self,
        X_train,
        Y_train,
        X_test,
        Y_test,
        base_learner_name,
        repeat_time,
        fold,
        dataset_name,
        noisy_rate,
    ):
        """Process CLR training and prediction."""
        results = []
        clr_time1 = time.time()
        clr = PredictBOPOs(base_classifier_name=base_learner_name.value)
        clr.fit_CLR(X_train, Y_train)
        log(INFO, f"CLR Training time: {(time.time() - clr_time1)} seconds")

        predict_time1 = time.time()
        predicted_Y, _ = clr.predict_CLR(X_test, Y_test.shape[1])
        log(INFO, f"CLR Prediction time {(time.time() - predict_time1)} seconds")

        save_time1 = time.time()
        self._update_results(
            results,
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
        log(INFO, f"CLR Saving time {(time.time() - save_time1)} seconds")

        return results

    def _process_br(
        self,
        X_train,
        Y_train,
        X_test,
        Y_test,
        base_learner_name,
        repeat_time,
        fold,
        dataset_name,
        noisy_rate,
    ):
        """Process BR training and prediction."""
        results = []
        br_time1 = time.time()
        br = PredictBOPOs(base_classifier_name=base_learner_name.value)
        br.fit_BR(X_train, Y_train)
        log(INFO, f"BR Training time: {(time.time() - br_time1)} seconds")

        predict_time1 = time.time()
        predicted_Y, _ = br.predict_BR(X_test, Y_test.shape[1])
        log(INFO, f"BR Prediction time {(time.time() - predict_time1)} seconds")

        save_time1 = time.time()
        self._update_results(
            results,
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
        log(INFO, f"BR Saving time {(time.time() - save_time1)} seconds")

        return results

    def _process_cc(
        self,
        X_train,
        Y_train,
        X_test,
        Y_test,
        base_learner_name,
        repeat_time,
        fold,
        dataset_name,
        noisy_rate,
    ):
        """Process CC training and prediction."""
        results = []
        cc_time1 = time.time()
        cc = PredictBOPOs(base_classifier_name=base_learner_name.value)
        cc.fit_CC(X_train, Y_train)
        log(INFO, f"CC Training time: {(time.time() - cc_time1)} seconds")

        predict_time1 = time.time()
        predicted_Y, _ = cc.predict_CC(X_test, Y_test.shape[1])
        log(INFO, f"CC Prediction time {(time.time() - predict_time1)} seconds")

        save_time1 = time.time()
        self._update_results(
            results,
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
        log(INFO, f"CC Saving time {(time.time() - save_time1)} seconds")

        return results

    def _update_results(
        self,
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
        """Update results with prediction data."""
        indices_vector = None
        if len(predict_results) == 3:
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

    def train(self, dataset_config: DatasetConfig):
        """Main training loop."""
        try:
            for noisy_rate in self.config.noisy_rates:
                log(INFO, f"Noisy rate: {noisy_rate}")

                for algorithm in self.config.algorithms:
                    log(INFO, f"Processing algorithm: {algorithm.value}")

                    # Process dataset for this algorithm
                    results = self.process_dataset(
                        0, noisy_rate, dataset_config.name, algorithm
                    )

                    # Save results immediately after processing
                    self._save_results(
                        dataset_config.name,
                        noisy_rate,
                        results,
                        algorithm,
                    )

        except Exception as e:
            log(ERROR, f"Training failed: {str(e)}")
            raise

    def _save_results(
        self,
        dataset_name: str,
        noisy_rate: float,
        results: List[Dict],
        algorithm: AlgorithmType,
    ):
        """Save results for a specific algorithm."""
        is_clr = algorithm == AlgorithmType.CLR
        is_br = algorithm == AlgorithmType.BR
        is_cc = algorithm == AlgorithmType.CC

        ExperimentResults.save_results(
            results,
            dataset_name,
            noisy_rate,
            self.config.results_dir,
            is_clr=is_clr,
            is_br=is_br,
            is_cc=is_cc,
        )
