import numpy as np
from scipy.io import arff
import pandas as pd


# Decrepated
def load_and_preprocess_data(file_path, data_file, n_labels):
    """Load and preprocess dataset from ARFF file."""
    data, meta = arff.loadarff(file_path + data_file)
    df = pd.DataFrame(data)

    if data_file in ["emotions.arff", "scene.arff"]:
        X = df.iloc[:, :-n_labels].to_numpy()
        Y = df.iloc[:, -n_labels:].to_numpy().astype(int)
    else:
        X = df.iloc[:, n_labels:].to_numpy()
        Y = df.iloc[:, :n_labels].to_numpy().astype(int)

    Y = np.where(Y < 0, 0, Y)
    return X, Y
