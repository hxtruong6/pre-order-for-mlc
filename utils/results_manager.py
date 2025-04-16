from dataclasses import dataclass
from logging import INFO, log
from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np
import ast

import logging


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

        # convert to int for Y_BOPOs
        df["Y_BOPOs"] = df["Y_BOPOs"].apply(lambda x: x.astype(int))

        return df


@dataclass
class ExperimentResults:
    """Class to handle experiment results and metrics."""

    @staticmethod
    def save_results(results, dataset_name, noisy_rate, results_dir, is_clr=False):
        """
        Saves the results dictionary to both pickle and CSV formats.

        Args:
            results: List of dictionaries containing experiment results
            dataset_name: Name of the dataset
            noisy_rate: Noise rate used in the experiment
        """
        # Create results directory if it doesn't exist
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Clean filename
        dataset_name = dataset_name.lower().replace(" ", "_")
        base_filename = f"{results_dir}/dataset_{dataset_name}_noisy_{noisy_rate}{'_clr' if is_clr else ''}"

        # Save as pickle for exact Python object preservation
        with open(f"{base_filename}.pkl", "wb") as f:
            pickle.dump(results, f)

        # Save as CSV for human readability and easy importing
        # df = pd.DataFrame(results)
        # df.to_csv(f"{base_filename}.csv", index=False)

        log(INFO, f"Results saved to {base_filename}.pkl and {base_filename}.csv")

    @staticmethod
    def load_results(path, dataset_name, noisy_rate, is_clr=False):
        """
        Loads results from pickle file.

        Args:
            dataset_name: Name of the dataset
            noisy_rate: Noise rate used in the experiment

        Returns:
            List of dictionaries containing experiment results
        """
        dataset_name = dataset_name.lower().replace(" ", "_")
        filename = f"{path}/dataset_{dataset_name}_noisy_{noisy_rate}{'_clr' if is_clr else ''}.pkl"

        log(INFO, f"Loading results from {filename}")

        with open(filename, "rb") as f:
            results = pickle.load(f)

        return ResultProcessor.process_predictions(pd.DataFrame(results))

    def to_dataframe(self) -> pd.DataFrame | None:
        """Convert results to pandas DataFrame."""

    def generate_summary(self) -> dict | None:
        """Generate summary statistics."""
