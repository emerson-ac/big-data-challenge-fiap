"""Shared utilities for the pipeline stages (seed, config, dataset hash)."""

import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
import yaml

from src.config import get_settings

logger = structlog.get_logger()


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
