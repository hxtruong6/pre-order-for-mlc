from dataclasses import dataclass
from logging import INFO, log
from pathlib import Path
import json
import pickle
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class ExperimentResults:
    """Class to handle experiment results and metrics."""

    @staticmethod
    def save_results(results, dataset_name, noisy_rate):
        """
        Saves the results dictionary to both pickle and CSV formats.

        Args:
            results: List of dictionaries containing experiment results
            dataset_name: Name of the dataset
            noisy_rate: Noise rate used in the experiment
        """
        # Create results directory if it doesn't exist
        Path("./results").mkdir(parents=True, exist_ok=True)

        # Clean filename
        dataset_name = dataset_name.lower().replace(" ", "_")
        base_filename = f"./results/dataset_{dataset_name}_noisy_{noisy_rate}"

        # Save as pickle for exact Python object preservation
        with open(f"{base_filename}.pkl", "wb") as f:
            pickle.dump(results, f)

        # Save as CSV for human readability and easy importing
        df = pd.DataFrame(results)
        df.to_csv(f"{base_filename}.csv", index=False)

        log(INFO, f"Results saved to {base_filename}.pkl and {base_filename}.csv")

    @staticmethod
    def load_results(path, dataset_name, noisy_rate):
        """
        Loads results from pickle file.

        Args:
            dataset_name: Name of the dataset
            noisy_rate: Noise rate used in the experiment

        Returns:
            List of dictionaries containing experiment results
        """
        dataset_name = dataset_name.lower().replace(" ", "_")
        filename = f"{path}/dataset_{dataset_name}_noisy_{noisy_rate}.pkl"

        log(INFO, f"Loading results from {filename}")

        with open(filename, "rb") as f:
            results = pickle.load(f)

        return results

    def to_dataframe(self) -> pd.DataFrame | None:
        """Convert results to pandas DataFrame."""

    def generate_summary(self) -> dict | None:
        """Generate summary statistics."""
