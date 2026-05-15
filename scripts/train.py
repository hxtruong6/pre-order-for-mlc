"""Entry point for training pairwise classifiers and inferring BOPOs.

Delegates to :class:`training_orchestrator.TrainingOrchestrator`, which is the
canonical training driver. Run-level configuration (datasets, noisy rates,
base learners, repeats, folds) is resolved through
:class:`config.ConfigManager`.
"""

import argparse
import time
from logging import INFO, basicConfig, log

from preorder4mlc.config import ConfigManager
from preorder4mlc.training_orchestrator import TrainingOrchestrator

basicConfig(level=INFO)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument(
        "--noise_rate",
        type=float,
        default=None,
        help=(
            "If set, override config.noisy_rates with [noise_rate] only "
            "(single-noise run for slurm splitting)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the training pipeline for one dataset."""
    args = parse_args()
    log(INFO, f"Arguments: {args}")

    config_manager = ConfigManager()
    dataset_config = config_manager.get_dataset_config(args.dataset)
    training_config = config_manager.get_training_config(args)

    orchestrator = TrainingOrchestrator(training_config)
    orchestrator.setup(dataset_config)

    start_time = time.time()
    orchestrator.train(dataset_config)
    log(INFO, f"Total time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
