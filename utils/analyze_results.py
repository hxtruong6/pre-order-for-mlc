import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from results_manager import ExperimentResults, ResultProcessor
import ast


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
