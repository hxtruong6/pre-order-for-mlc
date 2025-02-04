import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from results_manager import ExperimentResults
import ast


class ResultProcessor:
    """Class to handle processing of list-type columns in results DataFrame."""

    @staticmethod
    def convert_string_to_array(value: str) -> np.ndarray:
        """Convert string representation of list to numpy array."""
        if isinstance(value, str):
            return np.array(ast.literal_eval(value))
        return np.array(value)

    @staticmethod
    def process_predictions(df: pd.DataFrame) -> pd.DataFrame:
        """Process Y_predicted, Y_true, and Y_BOPOs columns."""
        # Convert string representations to numpy arrays if needed
        for col in ["Y_test", "Y_predicted", "Y_BOPOs"]:
            if col in df.columns:
                df[col] = df[col].apply(ResultProcessor.convert_string_to_array)  # type: ignore

        return df


class ResultAnalyzer:
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)

    def load_experiment_results(
        self, path: str, dataset_name: str, noisy_rate: float
    ) -> pd.DataFrame:
        """Load results for a specific dataset and noise rate."""
        results = ExperimentResults.load_results(path, dataset_name, noisy_rate)
        df = pd.DataFrame(results)
        df = ResultProcessor.process_predictions(df)

        print(df[["Y_test", "Y_predicted", "Y_BOPOs"]])

        print("Column type: ", df.dtypes)
        return df

    def convert_object_to_list(
        self, df: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """Convert an object column to a list of values."""
        return df[column_name].apply(lambda x: eval(x))  # type: ignore


def main():
    # Example usage
    analyzer = ResultAnalyzer()

    # Parameters
    dataset_name = "emotions"
    noisy_rate = 0.0
    path = "./results"
    # Load and analyze results
    results_df = analyzer.load_experiment_results(path, dataset_name, noisy_rate)


if __name__ == "__main__":
    main()
