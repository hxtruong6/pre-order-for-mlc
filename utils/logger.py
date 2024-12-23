import logging
from pathlib import Path


def setup_logger(experiment_name: str) -> logging.Logger:
    """Setup logger for the experiment"""
    logger = logging.getLogger(experiment_name)

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # File handler
    fh = logging.FileHandler(f"logs/{experiment_name}.log")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
