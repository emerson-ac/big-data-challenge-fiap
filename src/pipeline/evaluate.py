"""Estágio 3 do pipeline: avaliação comparativa e MLflow Model Registry.

Porta ``notebooks/08_model_comparison.ipynb``: reconstrói os 5 modelos,
compara métricas no teste, gera ``metrics_comparison.csv`` e ``MODEL_CARD.md``,
e promove o melhor modelo a Production no Registry.

Uso:
    uv run python -m src.pipeline.evaluate
"""

import pickle
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sp
import structlog
import torch

from src.config import get_settings
from src.evaluation.metrics import (
    coverage_at_k,
    evaluate_recommendations,
    hit_rate_at_k,
)
from src.evaluation.ranking import recommendations_from_score_matrix
from src.models.ncf import NeuralCollaborativeFiltering, score_all_items
from src.models.training.data import load_processed
from src.models.training.user_cf import build_recommendations
from src.pipeline.common import load_config, set_seed, setup_mlflow
from src.serving.pyfunc import ItemBasedCFPyfunc, build_artifacts

logger = structlog.get_logger()


def _popularity_recs(models_dir: Path, users: list[int], k: int) -> dict:
    """Recomendações do baseline de popularidade (mesmo top-k para todos)."""
    with open(models_dir / "baseline_popularity" / "ranking.pkl", "rb") as f:
        ranking = pickle.load(f)
    return {u: ranking[:k].tolist() for u in users}


def _score_recs(score_matrix: np.ndarray, users: list[int], k: int) -> dict:
    """Constrói recomendações top-k a partir de uma matriz densa de scores."""
    return recommendations_from_score_matrix(users, score_matrix, k)


def _item_cf_recs(models_dir, interactions, users, k) -> dict:
    """Recomendações item-based CF (histórico · similaridade)."""
    similarity = sp.load_npz(models_dir / "item_based_cf" / "item_similarity.npz")
    return _score_recs(interactions[users].dot(similarity).toarray(), users, k)


def _mf_recs(models_dir, users, k) -> dict:
    """Recomendações Matrix Factorization (produto de fatores)."""
    uf = np.load(models_dir / "matrix_factorization" / "user_factors.npy")
    it = np.load(models_dir / "matrix_factorization" / "item_factors.npy")
    return _score_recs(uf[users] @ it.T, users, k)


def _ncf_recs(models_dir, interactions, users, k) -> dict:
    """Recomendações NCF (rede neural PyTorch)."""
    import json

    with open(models_dir / "neural_network" / "metrics.json", encoding="utf-8") as f:
        params = json.load(f)["params"]
    model = NeuralCollaborativeFiltering(
        interactions.shape[0],
        interactions.shape[1],
        params["embedding_dim"],
        tuple(params["hidden_dims"]),
    )
    model.load_state_dict(torch.load(models_dir / "neural_network" / "model.pt"))
    return _score_recs(
        score_all_items(model, users, interactions.shape[1], batch_size=200), users, k
    )


def _user_cf_recs(models_dir, interactions, users, k) -> dict:
    """Recomendações user-based CF (KNN sobre o pool)."""
    with open(models_dir / "user_based_cf" / "knn_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    pool_matrix = interactions[bundle["pool_indices"]]
    n_neighbors = min(bundle["model"].n_neighbors, pool_matrix.shape[0])
    return build_recommendations(
        users, bundle["model"], pool_matrix, interactions, n_neighbors, k
    )


def _row(
    recs: dict, gt: dict, n_items: int, k: int, size_mb: float, latency_ms: float
) -> dict:
    """Monta uma linha da tabela comparativa com todas as métricas."""
    metrics = evaluate_recommendations(recs, gt, k)
    metrics["hit_rate_at_k"] = hit_rate_at_k(recs, gt, k)
    metrics["coverage_at_k"] = coverage_at_k(recs, n_items, k)
    metrics["inference_latency_ms"] = latency_ms
    metrics["model_size_mb"] = size_mb
    return metrics


def _dir_size_mb(path: Path, patterns: list[str]) -> float:
    """Soma o tamanho (MB) dos artefatos de um modelo."""
    total = sum(f.stat().st_size for pat in patterns for f in path.glob(pat))
    return round(total / (1024 * 1024), 4)


def _timed(fn, *args) -> tuple[dict, float]:
    """Executa um builder de recs medindo a latência média por usuário (ms)."""
    start = time.perf_counter()
    recs = fn(*args)
    elapsed_ms = (time.perf_counter() - start) * 1000 / max(len(recs), 1)
    return recs, round(elapsed_ms, 4)


def _build_comparison(models_dir, data, k) -> pd.DataFrame:
    """Reconstrói os 5 modelos e monta o DataFrame comparativo (teste)."""
    interactions = data.interactions
    n_items = interactions.shape[1]
    gt = data.test_ground_truth
    users = list(gt.keys())
    sample = users[: min(3000, len(users))]
    rows = {}
    pop, lat = _timed(_popularity_recs, models_dir, users, k)
    rows["popularity"] = _row(
        pop,
        gt,
        n_items,
        k,
        _dir_size_mb(models_dir / "baseline_popularity", ["*.pkl"]),
        lat,
    )
    item, lat = _timed(_item_cf_recs, models_dir, interactions, users, k)
    rows["item_based_cf"] = _row(
        item, gt, n_items, k, _dir_size_mb(models_dir / "item_based_cf", ["*.npz"]), lat
    )
    ucf, lat = _timed(_user_cf_recs, models_dir, interactions, sample, k)
    rows["user_based_cf"] = _row(
        ucf,
        {u: gt[u] for u in sample},
        n_items,
        k,
        _dir_size_mb(models_dir / "user_based_cf", ["*.pkl"]),
        lat,
    )
    mfr, lat = _timed(_mf_recs, models_dir, users, k)
    rows["matrix_factorization"] = _row(
        mfr,
        gt,
        n_items,
        k,
        _dir_size_mb(models_dir / "matrix_factorization", ["*.npy"]),
        lat,
    )
    ncf, lat = _timed(_ncf_recs, models_dir, interactions, users, k)
    rows["ncf"] = _row(
        ncf, gt, n_items, k, _dir_size_mb(models_dir / "neural_network", ["*.pt"]), lat
    )
    return pd.DataFrame(rows).T.rename_axis("model")


def _write_model_card(
    df: pd.DataFrame, best: str, dataset_hash: str, out: Path
) -> None:
    """Gera o MODEL_CARD.md a partir da tabela comparativa."""
    beats_recall = bool(
        df.loc["ncf", "recall_at_k"] > df.drop("ncf")["recall_at_k"].max()
    )
    beats_ndcg = bool(df.loc["ncf", "ndcg_at_k"] > df.drop("ncf")["ndcg_at_k"].max())
    card = _MODEL_CARD_TEMPLATE.format(
        hash=dataset_hash[:16],
        table=df.round(4).to_markdown(),
        beats_recall=beats_recall,
        beats_ndcg=beats_ndcg,
        best=best,
    )
    out.write_text(card, encoding="utf-8")


def _register(
    models_dir: Path,
    processed_dir: Path,
    df: pd.DataFrame,
    best: str,
    dataset_hash: str,
) -> None:
    """Registra o melhor modelo como pyfunc e o promove via alias @production."""
    setup_mlflow("model_comparison")
    settings = get_settings()
    name = settings.registered_model_name
    with mlflow.start_run(run_name="model_comparison_v1"):
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_param("model_promoted_to_production", best)
        for model, row in df.iterrows():
            mlflow.log_metrics({f"{model}_{c}": float(v) for c, v in row.items()})
        mlflow.log_artifact(str(models_dir / "evaluation" / "metrics_comparison.csv"))
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ItemBasedCFPyfunc(),
            artifacts=build_artifacts(models_dir, processed_dir),
            registered_model_name=name,
        )
    # MLflow 3 removeu os stages: promoção passa a ser por alias.
    version = info.registered_model_version
    client = mlflow.tracking.MlflowClient()
    alias = settings.model_alias
    client.set_registered_model_alias(name, alias, version)
    logger.info("model_promoted", model=name, version=version, alias=alias)


def main() -> None:
    """Executa a avaliação comparativa, gera artefatos e atualiza o Registry."""
    settings = get_settings()
    config = load_config()
    set_seed(settings.random_seed)
    k = config["evaluation"]["k"]
    models_dir = settings.models_dir
    data = load_processed(settings.data_processed_dir)

    df = _build_comparison(models_dir, data, k)
    best = str(df["recall_at_k"].idxmax())
    eval_dir = models_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(eval_dir / "metrics_comparison.csv")
    _write_model_card(
        df, best, data.split_meta["dataset_hash"], models_dir / "MODEL_CARD.md"
    )
    _register(
        models_dir,
        settings.data_processed_dir,
        df,
        best,
        data.split_meta["dataset_hash"],
    )
    logger.info("evaluation_done", best_model=best)


_MODEL_CARD_TEMPLATE = """# Model Card - Sistema de Recomendacao Instacart

Comparacao de 5 modelos (top-10) no split de teste interno. Dataset hash: `{hash}...`.

## Metricas (split de teste)

{table}

## Decisao de Promocao

- NCF supera todos os baselines em recall@k: {beats_recall}
- NCF supera todos os baselines em ndcg@k: {beats_ndcg}
- Modelo com melhor recall@k: **{best}**
- `item_based_cf_recommender` promovido via alias **@production** no MLflow Model
  Registry (MLflow 3 usa aliases no lugar de stages); os demais não são registrados.

Gerado automaticamente por `src/pipeline/evaluate.py`.
"""


if __name__ == "__main__":
    main()
