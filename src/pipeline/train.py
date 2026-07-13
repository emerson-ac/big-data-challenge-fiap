"""Pipeline stage 2: training of the 5 models.

Ports the logic of notebooks 03-07 into a DVC-reproducible script.
Reads ``data/processed/*`` and writes artifacts to ``models/<model>/``.

Usage:
    uv run python -m src.pipeline.train
"""

import structlog

from src.config import get_settings
from src.models.training import item_cf, mf, ncf_train, popularity, user_cf
from src.models.training.data import load_processed
from src.pipeline.common import load_config, set_seed, setup_mlflow

logger = structlog.get_logger()

RUN_NAMES = {
    "baseline_popularity": "baseline_popularity_v1",
    "item_based_cf": "item_based_cf_final",
    "user_based_cf": "user_based_cf_final",
    "matrix_factorization": "matrix_factorization_v1",
    "ncf": "ncf_final",
}


def _ctx(settings, config, k) -> dict:
    """Assembles the shared training context (paths, seed, k)."""
    return {
        "models_dir": settings.models_dir,
        "processed_dir": settings.processed_data_dir,
        "seed": settings.random_seed,
        "k": k,
        "config": config,
    }


def _mf(data, k, config, md, seed) -> dict:
    """Trains Matrix Factorization (kept apart to respect the line limit)."""
    return mf.train(
        data, k, config["matrix_factorization"], md / "matrix_factorization", seed
    )


def _train_each(data, ctx) -> tuple[str, dict]:
    """Yields (model_name, metrics_payload) for each of the 5 models in order."""
    md = ctx["models_dir"]
    seed = ctx["seed"]
    k = ctx["k"]
    config = ctx["config"]
    pdir = ctx["processed_dir"]
    yield "baseline_popularity", popularity.train(data, k, md / "baseline_popularity")
    yield (
        "item_based_cf",
        item_cf.train(data, k, config["item_based_cf"], md / "item_based_cf"),
    )
    yield (
        "user_based_cf",
        user_cf.train(data, k, config["user_based_cf"], md / "user_based_cf", seed),
    )
    yield "matrix_factorization", _mf(data, k, config, md, seed)
    yield (
        "ncf",
        ncf_train.train(data, k, config["ncf"], md / "neural_network", seed, pdir),
    )


def _log_run(name, payload, dataset_hash) -> None:
    """Logs a model's params and val/test metrics to MLflow."""
    import mlflow

    setup_mlflow(name)
    with mlflow.start_run(run_name=RUN_NAMES[name]):
        mlflow.log_param("k", payload["k"])
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_metrics({f"val_{m}": v for m, v in payload["validation"].items()})
        mlflow.log_metrics({f"test_{m}": v for m, v in payload["test"].items()})


def main() -> None:
    """Trains the 5 models in sequence and logs each result."""
    settings = get_settings()
    config = load_config()
    set_seed(settings.random_seed)
    data = load_processed(settings.processed_data_dir)
    dataset_hash = data.split_meta["dataset_hash"]
    ctx = _ctx(settings, config, config["evaluation"]["k"])
    for name, payload in _train_each(data, ctx):
        _log_run(name, payload, dataset_hash)
        logger.info(
            "model_trained",
            model=name,
            test_recall=payload["test"]["recall_at_k"],
        )


if __name__ == "__main__":
    main()
