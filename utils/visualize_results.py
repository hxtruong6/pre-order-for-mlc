import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
from pathlib import Path

# Configuration
SUMMARY_DIR = "results/final_0624_summary"
VISUALIZATION_DIR = f"{SUMMARY_DIR}/visualization"
if not os.path.exists(VISUALIZATION_DIR):
    os.makedirs(VISUALIZATION_DIR)


NOISE_LEVELS = ["0.0", "0.1", "0.2", "0.3"]
CHART_STYLES = {
    "figure.figsize": (12, 8),
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}

# Color palette for algorithms
COLOR_PALETTE = {
    "PreOrder__Hamming__None": "#1f77b4",
    "PreOrder__Hamming__2": "#ff7f0e",
    "PreOrder__Subset__None": "#2ca02c",
    "PreOrder__Subset__2": "#d62728",
    "PartialOrder__Hamming__None": "#9467bd",
    "PartialOrder__Hamming__2": "#8c564b",
    "PartialOrder__Subset__None": "#e377c2",
    "PartialOrder__Subset__2": "#7f7f7f",
    "br": "#bcbd22",
    "cc": "#17becf",
    "clr": "#ff9896",
}

# Metrics to focus on for visualization
KEY_METRICS = {
    "BinaryVector": ["hamming_accuracy", "f1", "subset0_1"],
    "PartialAbstention": [
        "hamming_accuracy_pa",
        "f1_pa",
        "subset0_1_pa",
        "arec",
        "aabs",
    ],
}


def extract_mean_from_string(value_str):
    """Extract mean value from 'mean±std' format."""
    if pd.isna(value_str) or value_str == "":
        return np.nan
    try:
        # Extract the mean part before the ± symbol
        mean_str = str(value_str).split("±")[0]
        return float(mean_str)
    except:
        return np.nan


def load_summary_data(summary_dir):
    """Load all summary CSV files and organize by dataset and prediction type."""
    summary_files = glob(os.path.join(summary_dir, "*_summary.csv"))
    data = {}

    print(f"Loading {len(summary_files)} summary files...")

    for file in summary_files:
        filename = os.path.basename(file)
        # Extract dataset and prediction type from filename
        match = re.match(
            r"(.+)_(BinaryVector|PartialAbstention)_summary\.csv", filename
        )
        if match:
            dataset = match.group(1)
            pred_type = match.group(2)

            print(f"Loading: {dataset} - {pred_type}")
            df = pd.read_csv(file)

            # Extract mean values from all metric columns
            for col in df.columns:
                if col != "Algorithm" and "__" in col:
                    df[col] = df[col].apply(extract_mean_from_string)

            data[(dataset, pred_type)] = df

    return data


def create_line_charts(data, output_dir):
    """Create line charts showing performance trends across noise levels."""
    print("\nCreating line charts...")

    for (dataset, pred_type), df in data.items():
        print(f"  Processing: {dataset} - {pred_type}")

        # Get key metrics for this prediction type
        metrics = KEY_METRICS.get(pred_type, [])

        # Create subplots for each metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Get columns for this metric across noise levels
            metric_cols = [f"{metric}__{noise}" for noise in NOISE_LEVELS]

            # Plot each algorithm
            for _, row in df.iterrows():
                algo = row["Algorithm"]
                values = [row[col] for col in metric_cols if not pd.isna(row[col])]

                if len(values) == len(
                    NOISE_LEVELS
                ):  # Only plot if we have all noise levels
                    color = COLOR_PALETTE.get(algo, "#000000")
                    ax.plot(
                        NOISE_LEVELS,
                        values,
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        label=algo,
                        color=color,
                    )

            ax.set_xlabel("Noise Level")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f'{metric.replace("_", " ").title()} - {dataset}')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        output_file = os.path.join(output_dir, f"{dataset}_{pred_type}_line_charts.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {output_file}")


def create_heatmaps(data, output_dir):
    """Create heatmaps comparing algorithms across noise levels for each metric."""
    print("\nCreating heatmaps...")

    for (dataset, pred_type), df in data.items():
        print(f"  Processing: {dataset} - {pred_type}")

        metrics = KEY_METRICS.get(pred_type, [])

        for metric in metrics:
            # Create pivot table for this metric
            metric_cols = [f"{metric}__{noise}" for noise in NOISE_LEVELS]

            # Prepare data for heatmap
            heatmap_data = []
            algorithms = []

            for _, row in df.iterrows():
                algo = row["Algorithm"]
                values = [row[col] for col in metric_cols]

                # Only include if we have valid data
                if not all(pd.isna(v) for v in values):
                    heatmap_data.append(values)
                    algorithms.append(algo)

            if heatmap_data:
                heatmap_df = pd.DataFrame(
                    heatmap_data, index=algorithms, columns=NOISE_LEVELS
                )

                # Create heatmap
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    heatmap_df,
                    annot=True,
                    fmt=".3f",
                    cmap="RdYlBu_r",
                    cbar_kws={"label": metric.replace("_", " ").title()},
                )
                plt.title(
                    f'{metric.replace("_", " ").title()} - {dataset} ({pred_type})'
                )
                plt.xlabel("Noise Level")
                plt.ylabel("Algorithm")

                output_file = os.path.join(
                    output_dir, f"{dataset}_{pred_type}_{metric}_heatmap.png"
                )
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"    Saved: {output_file}")


def create_bar_charts(data, output_dir):
    """Create bar charts comparing algorithms at specific noise levels."""
    print("\nCreating bar charts...")

    for (dataset, pred_type), df in data.items():
        print(f"  Processing: {dataset} - {pred_type}")

        metrics = KEY_METRICS.get(pred_type, [])

        # Create one chart per noise level
        for noise_level in NOISE_LEVELS:
            fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
            if len(metrics) == 1:
                axes = [axes]

            for i, metric in enumerate(metrics):
                ax = axes[i]
                metric_col = f"{metric}__{noise_level}"

                # Get data for this metric and noise level
                valid_data = []
                algorithms = []

                for _, row in df.iterrows():
                    algo = row["Algorithm"]
                    value = row[metric_col]

                    if not pd.isna(value):
                        valid_data.append(value)
                        algorithms.append(algo)

                if valid_data:
                    # Create bar chart
                    colors = [COLOR_PALETTE.get(algo, "#000000") for algo in algorithms]
                    bars = ax.bar(algorithms, valid_data, color=colors, alpha=0.8)

                    # Add value labels on bars
                    for bar, value in zip(bars, valid_data):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.01,
                            f"{value:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                    ax.set_xlabel("Algorithm")
                    ax.set_ylabel(metric.replace("_", " ").title())
                    ax.set_title(
                        f'{metric.replace("_", " ").title()} - Noise {noise_level}'
                    )
                    ax.tick_params(axis="x", rotation=45)
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = os.path.join(
                output_dir, f"{dataset}_{pred_type}_noise_{noise_level}_bar_charts.png"
            )
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"    Saved: {output_file}")


def create_summary_dashboard(data, output_dir):
    """Create a comprehensive dashboard with key insights."""
    print("\nCreating summary dashboard...")

    # Collect summary statistics
    summary_stats = []

    for (dataset, pred_type), df in data.items():
        metrics = KEY_METRICS.get(pred_type, [])

        for metric in metrics:
            metric_cols = [f"{metric}__{noise}" for noise in NOISE_LEVELS]

            # Find best performing algorithm at each noise level
            for i, noise in enumerate(NOISE_LEVELS):
                col = metric_cols[i]
                values = df[col].dropna()

                if not values.empty:
                    best_algo = df.loc[values.idxmax(), "Algorithm"]
                    best_value = values.max()
                    worst_algo = df.loc[values.idxmin(), "Algorithm"]
                    worst_value = values.min()

                    summary_stats.append(
                        {
                            "Dataset": dataset,
                            "Prediction_Type": pred_type,
                            "Metric": metric,
                            "Noise_Level": noise,
                            "Best_Algorithm": best_algo,
                            "Best_Value": best_value,
                            "Worst_Algorithm": worst_algo,
                            "Worst_Value": worst_value,
                            "Range": best_value - worst_value,
                        }
                    )

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_stats)

    # Save summary statistics
    summary_file = os.path.join(output_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"  Saved summary statistics: {summary_file}")

    # Create visualization of best algorithms
    if not summary_df.empty:
        # Best algorithm frequency
        best_algo_counts = summary_df["Best_Algorithm"].value_counts()

        plt.figure(figsize=(12, 8))
        best_algo_counts.plot(kind="bar", color="skyblue")
        plt.title("Frequency of Best Performing Algorithms Across All Metrics")
        plt.xlabel("Algorithm")
        plt.ylabel("Number of Times Best")
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_file = os.path.join(output_dir, "best_algorithms_summary.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_file}")


def main():
    """Main function to generate all visualizations."""
    print("=== Starting Results Visualization ===")

    # Set up matplotlib style
    plt.style.use("default")
    for key, value in CHART_STYLES.items():
        plt.rcParams[key] = value

    # Create output directory
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    print(f"Output directory: {VISUALIZATION_DIR}")

    # Load data
    data = load_summary_data(SUMMARY_DIR)

    if not data:
        print("No summary files found!")
        return

    print(f"Loaded data for {len(data)} dataset/prediction_type combinations")

    # Generate visualizations
    create_line_charts(data, VISUALIZATION_DIR)
    create_heatmaps(data, VISUALIZATION_DIR)
    create_bar_charts(data, VISUALIZATION_DIR)
    create_summary_dashboard(data, VISUALIZATION_DIR)

    print(f"\n=== Visualization Complete ===")
    print(f"All charts saved to: {VISUALIZATION_DIR}")


if __name__ == "__main__":
    main()
