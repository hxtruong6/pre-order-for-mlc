import os
import re
import pandas as pd
from glob import glob
from collections import defaultdict

RESULTS_DIR = "results/20250624"  # Updated to match the new folder
OUTPUT_DIR = "results/final_0624_summary"
NOISE_LEVELS = ["0.0", "0.1", "0.2", "0.3"]

# File patterns for each algorithm type
FILE_PATTERNS = {
    "bopos": "evaluation_*_noisy_*_bopos.csv",
    "br": "evaluation_*_noisy_*_br.csv",
    "cc": "evaluation_*_noisy_*_cc.csv",
    "clr": "evaluation_*_noisy_*_clr.csv",
}

# Prediction types to include for bopos files (skip PreferenceOrder)
BOPOS_PREDICTION_TYPES = ["BinaryVector", "PartialAbstention"]

# Helper to extract dataset and noise level from filename
dataset_regex = re.compile(r"evaluation_([A-Za-z0-9_]+)_noisy_")
NOISE_REGEX = re.compile(r"noisy_(\d\.\d)")


def find_files(results_dir):
    """Find all relevant evaluation files in the results directory."""
    files = []
    for key, pattern in FILE_PATTERNS.items():
        pattern_files = glob(os.path.join(results_dir, pattern))
        files.extend(pattern_files)
        print(
            f"Found {len(pattern_files)} files for {key}: {pattern_files[:3]}..."
        )  # Show first 3 files
    print(f"Total files found: {len(files)}")
    return sorted(files)


def extract_dataset(filename):
    """Extract dataset name from filename."""
    match = dataset_regex.search(filename)
    return match.group(1) if match else None


def extract_noise_level(filename):
    """Extract noise level as a string from filename."""
    match = NOISE_REGEX.search(filename)
    return match.group(1) if match else None


def format_mean_std(mean, std):
    """Format mean and std as 'mean±std' with rounding."""
    try:
        # mean = float(mean)
        # std = float(std)
        mean = round(mean, 4)
        std = round(std, 4)
        return f"{mean:.4f}±{std:.4f}"
    except Exception:
        return ""


def collect_metrics(files):
    """Collect all metrics for all algorithms and noise levels from the files, grouped by dataset and prediction type."""
    # Structure: dataset -> prediction_type -> metrics[algorithm][metric][noise] = (mean, std)
    dataset_metrics = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    dataset_all_metrics = defaultdict(lambda: defaultdict(set))

    print(f"\nProcessing {len(files)} files...")

    for file in files:
        dataset = extract_dataset(file)
        noise = extract_noise_level(file)
        if dataset is None or noise not in NOISE_LEVELS:
            print(f"Skipping file (invalid dataset/noise): {file}")
            continue

        print(
            f"Processing: {os.path.basename(file)} (dataset: {dataset}, noise: {noise})"
        )

        if file.endswith("_bopos.csv"):
            # Handle bopos files - filter by prediction type
            df = pd.read_csv(file)
            print(
                f"  BOPOS file: {len(df)} rows, prediction types: {df['Prediction_Type'].unique()}"
            )

            for _, row in df.iterrows():
                prediction_type = row["Prediction_Type"]

                # Skip PreferenceOrder prediction type for bopos
                if prediction_type not in BOPOS_PREDICTION_TYPES:
                    print(f"    Skipping PreferenceOrder: {prediction_type}")
                    continue

                algo = row["Algorithm"]
                metric = row["Metric"]
                mean = row["Mean"]
                std = row["Std"]

                dataset_metrics[dataset][prediction_type][algo][metric][noise] = (
                    mean,
                    std,
                )
                dataset_all_metrics[dataset][prediction_type].add(metric)
                print(
                    f"    Added: {prediction_type} - {algo} - {metric} - {noise}: {mean}±{std}"
                )

        else:
            # Handle br, cc, clr files - all prediction types included
            if file.endswith("_br.csv"):
                algo = "br"
            elif file.endswith("_cc.csv"):
                algo = "cc"
            elif file.endswith("_clr.csv"):
                algo = "clr"
            else:
                continue

            df = pd.read_csv(file)
            print(f"  {algo.upper()} file: {len(df)} rows")

            for _, row in df.iterrows():
                prediction_type = row["Prediction_Type"]
                metric = row["Metric"]
                mean = row["Mean"]
                std = row["Std"]

                dataset_metrics[dataset][prediction_type][algo][metric][noise] = (
                    mean,
                    std,
                )
                dataset_all_metrics[dataset][prediction_type].add(metric)
                print(
                    f"    Added: {prediction_type} - {algo} - {metric} - {noise}: {mean}±{std}"
                )

    # Convert metric sets to sorted lists
    sorted_metrics = {}
    for dataset in dataset_all_metrics:
        sorted_metrics[dataset] = {}
        for pred_type in dataset_all_metrics[dataset]:
            sorted_metrics[dataset][pred_type] = sorted(
                dataset_all_metrics[dataset][pred_type]
            )

    print(f"\nCollected metrics for {len(dataset_metrics)} datasets")
    for dataset, pred_types in dataset_metrics.items():
        print(f"  {dataset}: {list(pred_types.keys())}")

    return dataset_metrics, sorted_metrics


def build_summary_table(metrics, all_metrics):
    """Build a summary DataFrame with algorithms as rows and metric__noise as columns."""
    # Collect all algorithms
    algorithms = sorted(metrics.keys())
    print(f"  Building table with {len(algorithms)} algorithms: {algorithms}")

    columns = []
    for metric in all_metrics:
        for noise in NOISE_LEVELS:
            columns.append(f"{metric}__{noise}")
    print(f"  Creating {len(columns)} metric columns")

    data = []
    for algo in algorithms:
        row = []
        for metric in all_metrics:
            for noise in NOISE_LEVELS:
                val = metrics[algo][metric].get(noise, (None, None))
                if val[0] is not None and val[1] is not None:
                    row.append(format_mean_std(val[0], val[1]))
                else:
                    row.append("")
        data.append([algo] + row)

    df = pd.DataFrame(data, columns=["Algorithm"] + columns)
    print(f"  Created table: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def main():
    """Main function to generate per-dataset and per-prediction-type summary metrics tables."""
    print("=== Starting Metrics Summary Generation ===")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"BOPOS prediction types to include: {BOPOS_PREDICTION_TYPES}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Find all relevant files
    files = find_files(RESULTS_DIR)

    # Collect metrics grouped by dataset and prediction type
    dataset_metrics, dataset_all_metrics = collect_metrics(files)

    # Generate summary tables for each dataset and prediction type
    print(f"\n=== Generating Summary Tables ===")
    for dataset, pred_types in dataset_metrics.items():
        print(f"\nProcessing dataset: {dataset}")

        for prediction_type, metrics in pred_types.items():
            print(f"  Prediction type: {prediction_type}")

            all_metrics = dataset_all_metrics[dataset][prediction_type]
            summary_df = build_summary_table(metrics, all_metrics)

            # Create output filenames with prediction type
            output_csv = os.path.join(
                OUTPUT_DIR, f"{dataset}_{prediction_type}_summary.csv"
            )
            output_xlsx = os.path.join(
                OUTPUT_DIR, f"{dataset}_{prediction_type}_summary.xlsx"
            )

            # Save files
            summary_df.to_csv(output_csv, index=False)
            summary_df.to_excel(output_xlsx, index=False)
            print(f"    Saved: {output_csv} and {output_xlsx}")

    print(f"\n=== Summary Generation Complete ===")


if __name__ == "__main__":
    main()
