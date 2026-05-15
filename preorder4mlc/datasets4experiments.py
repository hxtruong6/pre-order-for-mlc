"""Dataset loading, splitting, and noise injection.

:class:`Datasets4Experiments` reads multi-label ARFF files, separates X
from Y based on whether labels are at the beginning or end of the file
(controlled by :data:`TARGET_IN_END_FILE_DATASETS`), and yields k-fold
splits with optional symmetric label-flip noise (Bernoulli with rate
``noisy_rate``) for the training fold of each split.
"""

from logging import INFO, log

import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.stats import bernoulli
from sklearn.model_selection import KFold

try:
    import arff as liac_arff  # liac-arff package — handles sparse ARFF
except ImportError:  # pragma: no cover
    liac_arff = None

TARGET_IN_END_FILE_DATASETS = [
    "emotions.arff",
    "scene.arff",
    "flags.arff",
    "VirusGO.arff",
    "VirusPseAAC.arff",
    "Yelp.arff",
    "birds.arff",
    "HumanPseAAC.arff",
    "PlantGO.arff",
    "GpositivePseAAC.arff",
    "PlantPseAAC.arff",
    # Large-K additions (COMETA convention: labels at end of attribute list).
    "CAL500.arff",
    "mediamill.arff",
    "bibtex.arff",
]


def _load_sparse_arff_to_dataframe(path: str) -> pd.DataFrame:
    """Parse sparse-format ARFF via liac-arff and densify to a DataFrame.

    Sparse rows look like ``{2 1, 5 1, ...}`` — every non-listed index is 0.
    liac-arff returns a (n_samples, n_attributes) list-of-lists with zeros
    filled in, which we wrap in a DataFrame keyed by the attribute names.
    """
    with open(path) as f:
        obj = liac_arff.load(f, return_type=liac_arff.DENSE_GEN)
        attr_names = [a[0] for a in obj["attributes"]]
        rows = list(obj["data"])
    return pd.DataFrame(rows, columns=attr_names)


class Datasets4Experiments:

    def __init__(self, data_path: str, data_files: list[dict]):
        self.data_path = data_path

        self.data_files = []
        self.n_labels_set = []
        for item in data_files:
            self.data_files.append(item["dataset_name"])
            self.n_labels_set.append(item["n_labels_set"])

        self.datasets: list[tuple[np.ndarray, np.ndarray, str]] = []

    def load_datasets(self):
        for file_name, n_labels in zip(self.data_files, self.n_labels_set):
            full_path = f"{self.data_path}{file_name}"
            log(INFO, f"Loading dataset from {full_path}")
            try:
                data, _meta = arff.loadarff(full_path)
                df = pd.DataFrame(data)
            except (ValueError, NotImplementedError) as e:
                # scipy.io.arff cannot parse sparse ARFF format
                # ({idx val, idx val, ...}). Fall back to liac-arff which
                # supports it, then densify.
                if liac_arff is None:
                    raise RuntimeError(
                        f"scipy.io.arff failed on {file_name} ({e}). "
                        "Install liac-arff (`pip install liac-arff`) to handle "
                        "sparse ARFF datasets like bibtex."
                    ) from e
                log(INFO, f"  scipy failed ({e!s}); retrying with liac-arff…")
                df = _load_sparse_arff_to_dataframe(full_path)

            is_target_in_end = any(
                f.lower() == file_name.lower() for f in TARGET_IN_END_FILE_DATASETS
            )

            X, Y = self.preprocess_data(df, n_labels, is_target_in_end)
            df_name = file_name.split(".")[0]
            self.datasets.append((X, Y, df_name))

    def preprocess_data(self, df, n_labels, is_target_in_end=False):
        if is_target_in_end:
            X = df.iloc[:, :-n_labels].to_numpy()
            Y = df.iloc[:, -n_labels:].to_numpy().astype(int)
        else:
            X = df.iloc[:, n_labels:].to_numpy()
            Y = df.iloc[:, :n_labels].to_numpy().astype(int)

        # liac-arff returns object dtype when attributes are stored as strings
        # in a sparse ARFF. Coerce to float for sklearn/LightGBM compatibility.
        if X.dtype == object:
            X = X.astype(float)

        # Map sklearn-style -1 (negative) labels to 0 so downstream pairwise
        # encoders see a clean {0,1} matrix.
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
