"""Utilitários compartilhados pelos estágios do pipeline (seed, config, MLflow)."""

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

# Namespace dos experimentos deste projeto no servidor MLflow (compartilhado).
EXPERIMENT_PREFIX = "recsys-instacart"


def set_seed(seed: int) -> None:
    """Fixa a seed de aleatoriedade (random, numpy, torch) para reprodutibilidade.

    Args:
        seed: Valor da semente a fixar.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config() -> dict[str, Any]:
    """Carrega o YAML de configuração de modelos.

    Returns:
        Dicionário com os hiperparâmetros do projeto.
    """
    with open(get_settings().config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_mlflow(experiment: str) -> None:
    """Configura o tracking URI e o experimento MLflow ativos.

    O experimento é prefixado com ``EXPERIMENT_PREFIX`` para não colidir com
    outros projetos no servidor MLflow compartilhado. O tracking URI também é
    exportado em ``MLFLOW_TRACKING_URI`` para ferramentas que leem a env nativa.

    Args:
        experiment: Nome do experimento (sem prefixo) a ativar.
    """
    uri = get_settings().mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_experiment(f"{EXPERIMENT_PREFIX}/{experiment}")


def compute_dataset_hash(paths: list[Path]) -> str:
    """Calcula o SHA256 combinado dos arquivos de dados brutos.

    Args:
        paths: Caminhos dos arquivos a incluir no hash.

    Returns:
        Hash SHA256 hexadecimal combinado.
    """
    hasher = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.hexdigest()
