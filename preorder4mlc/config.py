"""Run configuration for the BOPOs training pipeline.

Defines the typed config dataclasses (:class:`DatasetConfig`,
:class:`TrainingConfig`) and the per-run :class:`ConfigManager` factory
that maps a dataset key from the command line to its ARFF location, the
target label count, and the fixed list of noisy rates / base learners /
fold and repeat counts used throughout the paper.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from preorder4mlc.constants import BaseLearnerName


class AlgorithmType(Enum):
    BOPOS = "bopos"
    CLR = "clr"
    BR = "br"
    CC = "cc"
    ECC = "ecc"


@dataclass
class DatasetConfig:
    name: str
    file: str
    n_labels: int


@dataclass
class TrainingConfig:
    data_path: str
    results_dir: str
    noisy_rates: list[float]
    base_learners: list[BaseLearnerName]
    total_repeat_times: int
    number_folds: int
    algorithms: list[AlgorithmType]  # Which algorithms to run
    # When set, orchestrator skips repeats/folds whose index does not match.
    # Result pickle filename is suffixed with _r<repeat>_f<fold> to keep
    # split partials distinct; merge_split_results.py recombines them.
    repeat_idx_filter: int | None = None
    fold_idx_filter: int | None = None


class ConfigManager:
    DATASET_CONFIGS = {
        "chd_49": DatasetConfig("CHD_49", "CHD_49.arff", 6),
        "emotions": DatasetConfig("emotions", "emotions.arff", 6),
        "scene": DatasetConfig("scene", "scene.arff", 6),
        "yeast": DatasetConfig("Yeast", "Yeast.arff", 14),
        "water_quality": DatasetConfig("Water-quality", "Water-quality.arff", 14),
        "humanpseaac": DatasetConfig("HumanPseAAC", "HumanPseAAC.arff", 14),
        "gpositivepseaac": DatasetConfig("GpositivePseAAC", "GpositivePseAAC.arff", 4),
        "plantpseaac": DatasetConfig("PlantPseAAC", "PlantPseAAC.arff", 12),
        # enron (K=53). ARFF is NOT bundled — download from COMETA / MULAN
        # and place at ./data/enron.arff. See REPRODUCE.md.
        "enron": DatasetConfig("enron", "enron.arff", 53),
    }

    @staticmethod
    def get_dataset_config(dataset_name: str) -> DatasetConfig:
        dataset_name = dataset_name.lower()
        if dataset_name not in ConfigManager.DATASET_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} not found")
        return ConfigManager.DATASET_CONFIGS[dataset_name]

    @staticmethod
    def get_training_config(args) -> TrainingConfig:
        results_dir = args.results_dir if args.results_dir else "./results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # NOISY_RATES = [0.0, 0.1, 0.2, 0.3]
        # If --noise_rate is passed on the CLI, restrict to that single level
        # so the run can be split across slurm jobs by noise.
        single_noise = getattr(args, "noise_rate", None)
        if single_noise is not None:
            NOISY_RATES = [float(single_noise)]
        else:
            NOISY_RATES = [
                0.0,
                0.1,
                0.2,
                0.3,
            ]
        # Default is RF (paper-equivalent). Pass --base_learner LightGBM on
        # the CLI to switch to LightGBM (paper also reports LGBM results).
        BASE_LEARNERS = [BaseLearnerName.RF]
        bl_override = getattr(args, "base_learner", None)
        if bl_override:
            BASE_LEARNERS = [BaseLearnerName(bl_override)]
        ALGORITHMS = [
            AlgorithmType.BOPOS,
            AlgorithmType.CLR,
            AlgorithmType.BR,
            AlgorithmType.CC,
        ]
        algo_override = getattr(args, "algorithm", None)
        if algo_override:
            ALGORITHMS = [AlgorithmType(algo_override)]

        return TrainingConfig(
            data_path="./data/",
            results_dir=results_dir,
            noisy_rates=NOISY_RATES,
            base_learners=BASE_LEARNERS,
            total_repeat_times=5,
            number_folds=5,
            algorithms=ALGORITHMS,
            repeat_idx_filter=getattr(args, "repeat_idx", None),
            fold_idx_filter=getattr(args, "fold_idx", None),
        )
