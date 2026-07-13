"""Configuração central do projeto, externalizada via .env (Pydantic Settings)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Parâmetros de ambiente validados, lidos de variáveis/.env.

    Todas as variáveis usam o prefixo ``RECSYS_`` no ambiente
    (ex.: ``RECSYS_MODEL_TYPE=matrix_factorization``).
    """

    model_config = SettingsConfigDict(
        env_prefix="RECSYS_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    random_seed: int = 42
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    config_path: Path = PROJECT_ROOT / "configs" / "model_config.yaml"
    # MLflow externo por padrão (tech-challenger). Para dev offline, use um
    # backend sqlite (o MLflow 3 descontinuou o file store): sqlite:///mlflow.db
    mlflow_tracking_uri: str = Field(default="https://mlflow.pocsarcotech.com")
    # Modelo registrado + alias promovido (MLflow 3 usa aliases no lugar de stages).
    registered_model_name: str = "item_based_cf_recommender"
    model_alias: str = "production"
    # Fonte do modelo no serving: "registry" (MLflow) ou "local" (artefatos em disco).
    model_source: str = "registry"
    model_type: str = "item_based_cf"
    default_k: int = 10
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Retorna a instância única (cacheada) das configurações.

    Returns:
        Objeto Settings com os parâmetros de ambiente validados.
    """
    return Settings()
