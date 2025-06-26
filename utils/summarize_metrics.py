import os
import re
import pandas as pd
from glob import glob
from collections import defaultdict

RESULTS_DIR = "results/final_0623"
NOISE_LEVELS = ["0.0", "0.1", "0.2", "0.3"]

# File patterns for each algorithm type
FILE_PATTERNS = {
    "bopos": "evaluation_*_noisy_*_bopos.csv",
    "br": "evaluation_*_noisy_*_br.csv",
    "cc": "evaluation_*_noisy_*_cc.csv",
    "clr": "evaluation_*_noisy_*_clr.csv",
}

# Helper to extract dataset and noise level from filename
dataset_regex = re.compile(r"evaluation_([A-Za-z0-9]+)_noisy_")
NOISE_REGEX = re.compile(r"noisy_(\d\.\d)")


def find_files(results_dir):
    """Find all relevant evaluation files in the results directory."""
    files = []
    for key, pattern in FILE_PATTERNS.items():
        files.extend(glob(os.path.join(results_dir, pattern)))
    return files


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
        mean = float(mean)
        std = float(std)
        return f"{mean:.4f}±{std:.3f}"
    except Exception:
        return ""


def collect_metrics(files):
    """Collect all metrics for all algorithms and noise levels from the files, grouped by dataset."""
    # Structure: dataset -> metrics[algorithm][metric][noise] = (mean, std)
    dataset_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    dataset_all_metrics = defaultdict(set)
    for file in files:
        dataset = extract_dataset(file)
        noise = extract_noise_level(file)
        if dataset is None or noise not in NOISE_LEVELS:
            continue
        if file.endswith("_bopos.csv"):
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                algo = row["Algorithm"]
                metric = row["Metric"]
                mean = row["Mean"]
                std = row["Std"]
                dataset_metrics[dataset][algo][metric][noise] = (mean, std)
                dataset_all_metrics[dataset].add(metric)
        else:
            # br, cc, clr
            if file.endswith("_br.csv"):
                algo = "br"
            elif file.endswith("_cc.csv"):
                algo = "cc"
            elif file.endswith("_clr.csv"):
                algo = "clr"
            else:
                continue
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                metric = row["Metric"]
                mean = row["Mean"]
                std = row["Std"]
                dataset_metrics[dataset][algo][metric][noise] = (mean, std)
                dataset_all_metrics[dataset].add(metric)
    # Convert metric sets to sorted lists only when needed
    return dataset_metrics, {k: sorted(v) for k, v in dataset_all_metrics.items()}


def build_summary_table(metrics, all_metrics):
    """Build a summary DataFrame with algorithms as rows and metric__noise as columns."""
    # Collect all algorithms
    algorithms = sorted(metrics.keys())
    columns = []
    for metric in all_metrics:
        for noise in NOISE_LEVELS:
            columns.append(f"{metric}__{noise}")
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
    return df


def main():
    """Main function to generate per-dataset summary metrics tables."""
    files = find_files(RESULTS_DIR)
    dataset_metrics, dataset_all_metrics = collect_metrics(files)
    for dataset, metrics in dataset_metrics.items():
        print(f"Dataset: {dataset}")
        all_metrics = dataset_all_metrics[dataset]
        summary_df = build_summary_table(metrics, all_metrics)
        output_csv = os.path.join(RESULTS_DIR, f"{dataset}_summary.csv")
        output_xlsx = os.path.join(RESULTS_DIR, f"{dataset}_summary.xlsx")
        summary_df.to_csv(output_csv, index=False)
        summary_df.to_excel(output_xlsx, index=False)
        print(f"Summary table saved to {output_csv} and {output_xlsx}")


if __name__ == "__main__":
    main()
