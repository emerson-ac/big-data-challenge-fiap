"""Estágio 2 do pipeline: treino dos 5 modelos, rastreados no MLflow.

Porta a lógica dos notebooks 03-07 para um script reproduzível via DVC.
Lê ``data/processed/*`` e escreve os artefatos em ``models/<modelo>/``.

Uso:
    uv run python -m src.pipeline.train
"""

import mlflow
import structlog

from src.config import get_settings
from src.models.training import item_cf, mf, ncf_train, popularity, user_cf
from src.models.training.data import load_processed
from src.pipeline.common import load_config, set_seed, setup_mlflow

logger = structlog.get_logger()


def _log_run(experiment: str, run_name: str, payload: dict, dataset_hash: str) -> None:
    """Rastreia params e métricas val/test de um modelo no MLflow.

    Args:
        experiment: Nome do experimento MLflow.
        run_name: Nome da run.
        payload: metrics.json do modelo (contém validation e test).
        dataset_hash: Hash do dataset, exigido em toda run.
    """
    setup_mlflow(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("k", payload["k"])
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_metrics({f"val_{m}": v for m, v in payload["validation"].items()})
        mlflow.log_metrics({f"test_{m}": v for m, v in payload["test"].items()})


def main() -> None:
    """Treina os 5 modelos em sequência e rastreia cada um no MLflow."""
    settings = get_settings()
    config = load_config()
    set_seed(settings.random_seed)
    k = config["evaluation"]["k"]
    seed = settings.random_seed
    models_dir = settings.models_dir
    data = load_processed(settings.data_processed_dir)
    dataset_hash = data.split_meta["dataset_hash"]

    runs = [
        (
            "baseline_popularity",
            "baseline_popularity_v1",
            popularity.train(data, k, models_dir / "baseline_popularity"),
        ),
        (
            "item_based_cf",
            "item_based_cf_final",
            item_cf.train(
                data, k, config["item_based_cf"], models_dir / "item_based_cf"
            ),
        ),
        (
            "user_based_cf",
            "user_based_cf_final",
            user_cf.train(
                data, k, config["user_based_cf"], models_dir / "user_based_cf", seed
            ),
        ),
        (
            "matrix_factorization",
            "matrix_factorization_v1",
            mf.train(
                data,
                k,
                config["matrix_factorization"],
                models_dir / "matrix_factorization",
                seed,
            ),
        ),
        (
            "ncf",
            "ncf_final",
            ncf_train.train(
                data,
                k,
                config["ncf"],
                models_dir / "neural_network",
                seed,
                settings.data_processed_dir,
            ),
        ),
    ]
    for experiment, run_name, payload in runs:
        _log_run(experiment, run_name, payload, dataset_hash)
        logger.info(
            "model_trained",
            model=experiment,
            test_recall=payload["test"]["recall_at_k"],
        )


if __name__ == "__main__":
    main()
