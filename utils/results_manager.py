from dataclasses import dataclass
from logging import INFO, log
from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np
import ast

import logging

from config import AlgorithmType


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
    def save_results(
        results,
        dataset_name,
        noisy_rate,
        results_dir,
        is_clr=False,
        is_br=False,
        is_cc=False,
    ):
        """
        Saves the results dictionary to both pickle and CSV formats.

        Args:
            results: List of dictionaries containing experiment results
            dataset_name: Name of the dataset
            noisy_rate: Noise rate used in the experiment
            results_dir: Directory to save results
            is_clr: Whether results are from CLR
            is_br: Whether results are from Binary Relevance
            is_cc: Whether results are from Classifier Chain
        """
        # Create results directory if it doesn't exist
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Clean filename
        dataset_name = dataset_name.lower().replace(" ", "_")

        # Determine suffix based on method type
        suffix = ""
        if is_clr:
            suffix = "_clr"
        elif is_br:
            suffix = "_br"
        elif is_cc:
            suffix = "_cc"

        base_filename = (
            f"{results_dir}/dataset_{dataset_name}_noisy_{noisy_rate}{suffix}"
        )

        # Save as pickle for exact Python object preservation
        with open(f"{base_filename}.pkl", "wb") as f:
            pickle.dump(results, f)

        # Save as CSV for human readability and easy importing
        # df = pd.DataFrame(results)
        # df.to_csv(f"{base_filename}.csv", index=False)

        log(INFO, f"Results saved to {base_filename}.pkl and {base_filename}.csv")

    @staticmethod
    def load_results(
        path,
        dataset_name,
        noisy_rate,
        algorithm_type: AlgorithmType = AlgorithmType.BOPOS,
    ):
        """
        Loads results from pickle file.

        Args:
            path: Path to results directory
            dataset_name: Name of the dataset
            noisy_rate: Noise rate used in the experiment
            is_clr: Whether results are from CLR
            is_br: Whether results are from Binary Relevance
            is_cc: Whether results are from Classifier Chain

        Returns:
            List of dictionaries containing experiment results
        """
        dataset_name = dataset_name.lower().replace(" ", "_")

        # Determine suffix based on method type
        suffix = ""  # default is BOPOs
        if algorithm_type == AlgorithmType.CLR:
            suffix = "_clr"
        elif algorithm_type == AlgorithmType.BR:
            suffix = "_br"
        elif algorithm_type == AlgorithmType.CC:
            suffix = "_cc"

        filename = f"{path}/dataset_{dataset_name}_noisy_{noisy_rate}{suffix}.pkl"

        log(INFO, f"Loading results from {filename}")

        with open(filename, "rb") as f:
            results = pickle.load(f)

        return ResultProcessor.process_predictions(pd.DataFrame(results))

    def to_dataframe(self) -> pd.DataFrame | None:
        """Convert results to pandas DataFrame."""

    def generate_summary(self) -> dict | None:
        """Generate summary statistics."""
