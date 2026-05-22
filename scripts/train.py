"""Entry point for training pairwise classifiers and inferring BOPOs.

Delegates to :class:`training_orchestrator.TrainingOrchestrator`, which is the
canonical training driver. Run-level configuration (datasets, noisy rates,
base learners, repeats, folds) is resolved through
:class:`config.ConfigManager`.
"""

import argparse
import os
import time
from logging import INFO, basicConfig, log

from preorder4mlc.config import AlgorithmType, ConfigManager
from preorder4mlc.constants import BaseLearnerName
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
    parser.add_argument(
        "--base_learner",
        type=str,
        default=None,
        choices=[b.value for b in BaseLearnerName],
        help="If set, override config.BASE_LEARNERS with this single learner.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        choices=["glpk", "highs"],
        help="MILP solver backend for the BOPOs ILP. Default: glpk.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=[a.value for a in AlgorithmType],
        help=(
            "If set, override config.ALGORITHMS with this single algorithm "
            "(single-algorithm run for slurm splitting)."
        ),
    )
    parser.add_argument(
        "--repeat_idx",
        type=int,
        default=None,
        help="If set, run only this repeat index (0-based, single-repeat split).",
    )
    parser.add_argument(
        "--fold_idx",
        type=int,
        default=None,
        help="If set, run only this fold index (0-based, single-fold split).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the training pipeline for one dataset."""
    args = parse_args()
    if args.solver:
        os.environ["PREORDER_SOLVER"] = args.solver
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
