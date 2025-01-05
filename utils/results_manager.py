from dataclasses import dataclass
from logging import INFO, log
from pathlib import Path
import json
import pickle
import pandas as pd


@dataclass
class ExperimentResults:
    """Class to handle experiment results and metrics."""

    @staticmethod
    def save_results(results, dataset_name, noisy_rate):
        """
        Saves the results dictionary to a JSON file for easy reloading.
        """
        dataset_name = dataset_name.lower().replace(" ", "_")
        filename = f"./results/new/dataset_{dataset_name}__noisy_{noisy_rate}"
        # with open(f"{filename}.json", "w") as f:
        #     json.dump(results, f, indent=4)

        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(results, f)

        with open(f"{filename}.txt", "w") as f:
            f.write(str(results))

        log(INFO, f"Results saved to {filename}")

    def load_results(self, path: Path) -> None:
        """Load results from specified path."""

    def to_dataframe(self) -> pd.DataFrame | None:
        """Convert results to pandas DataFrame."""

    def generate_summary(self) -> dict | None:
        """Generate summary statistics."""
