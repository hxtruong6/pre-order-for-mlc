from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd


@dataclass
class ExperimentResults:
    """Class to handle experiment results and metrics."""

    def save_results(self, path: Path) -> None:
        """Save results to specified path."""

    def load_results(self, path: Path) -> None:
        """Load results from specified path."""

    def to_dataframe(self) -> pd.DataFrame | None:
        """Convert results to pandas DataFrame."""

    def generate_summary(self) -> dict | None:
        """Generate summary statistics."""
