import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import KFold
from scipy.stats import bernoulli
from sklearn.preprocessing import StandardScaler

TARGET_IN_END_FILE_DATASETS = ["emotions.arff", "scene.arff"]


class Datasets4Experiments:
    def __init__(self, data_path, data_files, n_labels_set):
        self.data_path = data_path
        self.data_files = data_files
        self.n_labels_set = n_labels_set
        self.datasets = []

    def load_datasets(self):
        for file_name, n_labels in zip(self.data_files, self.n_labels_set):
            full_path = f"{self.data_path}{file_name}"
            data, meta = arff.loadarff(full_path)
            df = pd.DataFrame(data)
            X, Y = self.preprocess_data(
                df, n_labels, is_target_in_end=file_name in TARGET_IN_END_FILE_DATASETS
            )
            df_name = file_name.split(".")[0]
            self.datasets.append((X, Y, df_name))

    def preprocess_data(self, df, n_labels, is_target_in_end=False):
        if is_target_in_end:
            X = df.iloc[:, :-n_labels].to_numpy()
            Y = df.iloc[:, -n_labels:].to_numpy().astype(int)
        else:
            X = df.iloc[:, n_labels:].to_numpy()
            Y = df.iloc[:, :n_labels].to_numpy().astype(int)

        # TODO: Add preprocessing steps here
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)

        Y = np.where(Y < 0, 0, Y)

        return X, Y

    def add_noise_to_labels(self, Y, noisy_rate):
        """
        Adds noise to the dataset labels based on the specified noisy rate.

        :param Y: The label matrix for a dataset.
        :param noisy_rate: The rate at which noise should be added to the labels.
        :return: The label matrix with added noise.
        """
        n_instances, n_labels = Y.shape
        for i in range(n_instances):
            for j in range(n_labels):
                if bernoulli.rvs(p=noisy_rate):
                    Y[i, j] = 1 - Y[i, j]  # Flip the label to add noise
        return Y

    def kfold_split_with_noise(
        self, dataset_index, n_splits=5, noisy_rate=0.0, random_state=None, shuffle=True
    ):
        """
        Generates K-fold splits for a specific dataset and adds noise to the training labels.

        :param dataset_index: Index of the dataset to split.
        :param n_splits: Number of folds.
        :param noisy_rate: Noise rate to be applied to the training set labels.
        :param random_state: Random state for reproducibility.
        :param shuffle: Whether to shuffle the data before splitting.
        :return: Generator of K-fold splits (train_index, test_index) with noisy training labels.
        """
        X, Y, _ = self.datasets[dataset_index]
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        for train_index, test_index in kf.split(X):
            Y_train_noisy = self.add_noise_to_labels(Y[train_index].copy(), noisy_rate)
            # I want to return x_train, y_train_noisy, x_test, y_test
            yield X[train_index], Y_train_noisy, X[test_index], Y[test_index]

    def get_datasets(self) -> list:
        return self.datasets

    def get_length(self) -> int:
        return len(self.datasets)

    def get_dataset_name(self, dataset_index) -> str:
        return self.datasets[dataset_index][2]