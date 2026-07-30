"""Application settings externalized via environment variables (.env).

Validated with Pydantic Settings so misconfiguration fails fast at startup
instead of producing silent runtime errors (reproducibility, Aula 03).
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

RANDOM_SEED = 42
REQUIRED_PACKAGES = ("torch", "sklearn", "mlflow", "dvc", "pandas", "numpy")


class Settings(BaseSettings):
    """Central, validated configuration loaded from a .env file.

    Attributes:
        random_seed: Single source of truth for every stochastic operation.
        raw_data_dir: Location of the original, unprocessed dataset.
        processed_data_dir: Location of the preprocessed artifacts.
        models_dir: Location of the trained model artifacts.
        config_path: Path to the YAML hyperparameter file.
        mlflow_tracking_uri: MLflow tracking server address.
        mlflow_experiment_name: Default MLflow experiment name.
        registered_model_name: Model-agnostic name of the promoted-model
            container in the MLflow Model Registry (holds the round's best model).
        model_alias: Alias for the promoted model (e.g. "production").
        model_source: Where the API loads the model from ("local" or "registry").
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    random_seed: int = RANDOM_SEED
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    config_path: Path = Path("configs/model_config.yaml")
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "tech-challenge-recsys"
    registered_model_name: str = "recsys_recommender"
    model_alias: str = "production"
    model_source: str = "local"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached, validated Settings instance."""
    return Settings()


def check_environment() -> list[tuple[str, bool, str]]:
    """Runs reproducibility and dependency checks, returning their results.

    Returns:
        List of (check_name, passed, detail) tuples.
    """
    settings = get_settings()
    checks: list[tuple[str, bool, str]] = []
    checks.append(_check_python())
    checks.extend(_check_packages())
    checks.append(_check_seed(settings))
    checks.append(_check_config_file(settings))
    return checks


def _check_python() -> tuple[str, bool, str]:
    """Checks that the Python version meets the project requirement."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    passed = sys.version_info >= (3, 12)
    return ("python_version", passed, f"{version} (requires >=3.12)")


def _check_packages() -> list[tuple[str, bool, str]]:
    """Checks that every required runtime package is importable."""
    results: list[tuple[str, bool, str]] = []
    for package in REQUIRED_PACKAGES:
        passed, detail = _try_import(package)
        results.append((f"import_{package}", passed, detail))
    return results


def _try_import(package: str) -> tuple[bool, str]:
    """Attempts to import a package and reports its version when available."""
    try:
        module: Any = importlib.import_module(package)
    except ImportError as error:
        return (False, f"import failed: {error}")
    version = getattr(module, "__version__", "unknown")
    return (True, f"v{version}")


def _check_seed(settings: Settings) -> tuple[str, bool, str]:
    """Checks that the configured seed matches the project standard."""
    passed = settings.random_seed == RANDOM_SEED
    return ("random_seed", passed, f"{settings.random_seed} (expected {RANDOM_SEED})")


def _check_config_file(settings: Settings) -> tuple[str, bool, str]:
    """Checks that the hyperparameter config file is present."""
    exists = settings.config_path.exists()
    detail = str(settings.config_path) if exists else "file not found"
    return ("config_path", exists, detail)
