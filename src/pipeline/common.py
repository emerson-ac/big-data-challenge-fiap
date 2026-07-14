"""Shared utilities for the pipeline stages (seed, config, dataset hash, MLflow)."""

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import structlog
import torch
import yaml

from src.config import get_settings

logger = structlog.get_logger()

EXPERIMENT_PREFIX = "recsys-instacart"


def set_seed(seed: int) -> None:
    """Fixes the random seed (random, numpy, torch) for reproducibility.

    Args:
        seed: The seed value to fix.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config() -> dict[str, Any]:
    """Loads the model configuration YAML.

    Returns:
        Dictionary with the project hyperparameters.
    """
    with open(get_settings().config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_mlflow(experiment: str) -> None:
    """Sets the tracking URI and active MLflow experiment.

    The experiment is prefixed with ``EXPERIMENT_PREFIX`` to avoid colliding
    with other projects on a shared MLflow server. The tracking URI is also
    exported to ``MLFLOW_TRACKING_URI`` for tools reading the native env.

    Args:
        experiment: Experiment name (without prefix) to activate.
    """
    uri = get_settings().mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_experiment(f"{EXPERIMENT_PREFIX}/{experiment}")


def compute_dataset_hash(paths: list[Path]) -> str:
    """Computes the combined SHA256 of the raw data files.

    Args:
        paths: Paths of the files to include in the hash.

    Returns:
        Combined hex SHA256 hash.
    """
    hasher = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.hexdigest()
