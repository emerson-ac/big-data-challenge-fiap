"""Pipeline stage 3: comparative evaluation and model card generation.

Ports ``notebooks/08_model_comparison.ipynb``: rebuilds the 5 models, compares
metrics on the test split, and writes ``metrics_comparison.csv`` plus
``MODEL_CARD.md``.

Usage:
    uv run python -m src.pipeline.evaluate
"""

import json
import time
from pathlib import Path

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
from src.evaluation.promotion import select_promoted_model
from src.evaluation.ranking import recommendations_from_score_matrix
from src.models.ncf import NeuralCollaborativeFiltering, score_all_items
from src.models.training.data import load_processed
from src.models.training.user_cf import build_recommendations
from src.pipeline.common import load_config, set_seed, setup_mlflow
from src.serving.pyfunc import RecommenderPyfunc, build_artifacts

logger = structlog.get_logger()


def _score_recs(score_matrix: np.ndarray, users: list[int], k: int) -> dict:
    """Builds top-k recommendations from a dense score matrix."""
    return recommendations_from_score_matrix(users, score_matrix, k)


def _popularity_recs(models_dir: Path, users: list[int], k: int) -> dict:
    """Popularity baseline recommendations (same top-k for everyone)."""
    import pickle

    with open(models_dir / "baseline_popularity" / "ranking.pkl", "rb") as f:
        ranking = pickle.load(f)
    return {u: ranking[:k].tolist() for u in users}


def _item_cf_recs(models_dir, interactions, users, k) -> dict:
    """Item-based CF recommendations (history dot similarity)."""
    similarity = sp.load_npz(models_dir / "item_based_cf" / "item_similarity.npz")
    return _score_recs(interactions[users].dot(similarity).toarray(), users, k)


def _mf_recs(models_dir, users, k) -> dict:
    """Matrix Factorization recommendations (factor product)."""
    uf = np.load(models_dir / "matrix_factorization" / "user_factors.npy")
    it = np.load(models_dir / "matrix_factorization" / "item_factors.npy")
    return _score_recs(uf[users] @ it.T, users, k)


def _user_cf_recs(models_dir, interactions, users, k) -> dict:
    """User-based CF recommendations (KNN over the pool)."""
    import pickle

    with open(models_dir / "user_based_cf" / "knn_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    pool_matrix = interactions[bundle["pool_indices"]]
    n_neighbors = min(bundle["model"].n_neighbors, pool_matrix.shape[0])
    return build_recommendations(
        users, bundle["model"], pool_matrix, interactions, n_neighbors, k
    )


def _ncf_recs(models_dir, interactions, users, k) -> dict:
    """NCF recommendations (PyTorch neural network)."""
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


def _dir_size_mb(path: Path, patterns: list[str]) -> float:
    """Sums the size (MB) of a model's artifacts."""
    total = sum(f.stat().st_size for pat in patterns for f in path.glob(pat))
    return round(total / (1024 * 1024), 4)


def _timed(fn, *args) -> tuple[dict, float]:
    """Runs a recs builder measuring the average per-user latency (ms)."""
    start = time.perf_counter()
    recs = fn(*args)
    elapsed_ms = (time.perf_counter() - start) * 1000 / max(len(recs), 1)
    return recs, round(elapsed_ms, 4)


def _row(
    recs: dict, gt: dict, n_items: int, k: int, size_mb: float, latency_ms: float
) -> dict:
    """Assembles a comparison-table row with all metrics."""
    metrics = evaluate_recommendations(recs, gt, k)
    metrics["hit_rate_at_k"] = hit_rate_at_k(recs, gt, k)
    metrics["coverage_at_k"] = coverage_at_k(recs, n_items, k)
    metrics["inference_latency_ms"] = latency_ms
    metrics["model_size_mb"] = size_mb
    return metrics


def _model_row(recs, gt, n_items, k, model_dir, patterns, lat) -> dict:
    """Assembles a comparison row from recs, artifact size and latency."""
    return _row(recs, gt, n_items, k, _dir_size_mb(model_dir, patterns), lat)


def _sample_users(users: list[int], cap: int, seed: int) -> list[int]:
    """Selects a reproducible, uniform evaluation population.

    Returns all users when the test split fits under the cap; otherwise a
    seeded random sample (no user_idx bias), shared by every model so the
    comparison table stays apples-to-apples (same population, same coverage
    denominator context).

    Args:
        users: All test-split user indices.
        cap: Maximum evaluation population size.
        seed: RNG seed for reproducibility.

    Returns:
        The evaluation user indices (sorted for determinism).
    """
    if len(users) <= cap:
        return users
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(users), size=cap, replace=False)
    return [users[i] for i in sorted(chosen)]


def _add_row(rows, name, recs_fn, gt, ctx):
    """Times a recs builder and appends its comparison row to ``rows``."""
    sub, patterns = _MODEL_DIRS[name]
    recs, lat = _timed(recs_fn)
    md = ctx["md"]
    rows[name] = _model_row(recs, gt, ctx["n_items"], ctx["k"], md / sub, patterns, lat)


_MODEL_DIRS = {
    "popularity": ("baseline_popularity", ["*.pkl"]),
    "item_based_cf": ("item_based_cf", ["*.npz"]),
    "user_based_cf": ("user_based_cf", ["*.pkl"]),
    "matrix_factorization": ("matrix_factorization", ["*.npy"]),
    "ncf": ("neural_network", ["*.pt"]),
}


def _collect_rows(models_dir, interactions, users, full_gt, k) -> dict:
    """Rebuilds the 5 models and scores them on a single shared population.

    Every model is evaluated on the same ``users`` (and their ground truth),
    so all metrics — including coverage — are directly comparable.
    """
    n_items = interactions.shape[1]
    gt = {u: full_gt[u] for u in users}
    md, it = models_dir, interactions
    ctx = {"md": md, "n_items": n_items, "k": k}
    rows: dict[str, dict] = {}
    _add_row(rows, "popularity", lambda: _popularity_recs(md, users, k), gt, ctx)
    _add_row(rows, "item_based_cf", lambda: _item_cf_recs(md, it, users, k), gt, ctx)
    _add_row(rows, "user_based_cf", lambda: _user_cf_recs(md, it, users, k), gt, ctx)
    _add_row(rows, "matrix_factorization", lambda: _mf_recs(md, users, k), gt, ctx)
    _add_row(rows, "ncf", lambda: _ncf_recs(md, it, users, k), gt, ctx)
    return rows


def _ncf_verdict(beats_recall: bool, beats_ndcg: bool) -> str:
    """One-line verdict on how the NCF compared to the CF baselines."""
    if beats_recall and beats_ndcg:
        return "superou os baselines de CF em recall@k e ndcg@k"
    if beats_recall or beats_ndcg:
        return "superou parcialmente os baselines de CF (ver tabela)"
    return "nao superou os baselines de CF neste dataset"


def _write_model_card(
    df: pd.DataFrame, dataset_hash: str, out: Path, ctx: dict
) -> None:
    """Generates MODEL_CARD.md (performance, limitations and biases)."""
    beats_recall = bool(
        df.loc["ncf", "recall_at_k"] > df.drop("ncf")["recall_at_k"].max()
    )
    beats_ndcg = bool(df.loc["ncf", "ndcg_at_k"] > df.drop("ncf")["ndcg_at_k"].max())
    card = _MODEL_CARD_TEMPLATE.format(
        hash=dataset_hash[:16],
        table=df.round(4).to_markdown(),
        beats_recall=beats_recall,
        beats_ndcg=beats_ndcg,
        ncf_verdict=_ncf_verdict(beats_recall, beats_ndcg),
        best=ctx["best"],
        best_cov=round(float(df.loc[ctx["best"], "coverage_at_k"]), 4),
        registered_name=ctx["registered_name"],
        n_eval=ctx["n_eval"],
        k=ctx["k"],
        top_n=ctx["top_n"],
    )
    out.write_text(card, encoding="utf-8")


def _log_comparison(df, best, dataset_hash, models_dir) -> None:
    """Logs comparison metrics and the comparison CSV to the active run."""
    import mlflow

    mlflow.log_param("dataset_hash", dataset_hash)
    mlflow.log_param("model_promoted_to_production", best)
    for model, row in df.iterrows():
        mlflow.log_metrics({f"{model}_{c}": float(v) for c, v in row.items()})
    mlflow.log_artifact(str(models_dir / "evaluation" / "metrics_comparison.csv"))


def _register(df, best, dataset_hash, settings) -> None:
    """Logs comparison metrics and registers the best model as a pyfunc."""
    import mlflow

    setup_mlflow("model_comparison")
    name = settings.registered_model_name
    with mlflow.start_run(run_name="model_comparison_v1"):
        _log_comparison(df, best, dataset_hash, settings.models_dir)
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RecommenderPyfunc(best),
            artifacts=build_artifacts(
                best, settings.models_dir, settings.processed_data_dir
            ),
            registered_model_name=name,
            input_example=pd.DataFrame({"user_idx": [1], "k": [10]}),
        )
    _promote(name, info.registered_model_version, settings.model_alias)


def _promote(name: str, version: str, alias: str) -> None:
    """Promotes the model through Staging -> Production stages + alias."""
    import mlflow

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(name, version, "Staging")
    client.transition_model_version_stage(name, version, "Production")
    client.set_registered_model_alias(name, alias, version)
    logger.info(
        "model_promoted",
        model=name,
        version=version,
        stages=["Staging", "Production"],
        alias=alias,
    )


def _card_context(settings, config, best: str, n_eval: int) -> dict:
    """Assembles the dynamic values injected into the MODEL_CARD."""
    return {
        "registered_name": settings.registered_model_name,
        "best": best,
        "n_eval": n_eval,
        "k": config["evaluation"]["k"],
        "top_n": config["preprocessing"]["top_n_products"],
    }


def _save_artifacts(df, data, models_dir, ctx) -> None:
    """Writes the comparison CSV and the dynamic MODEL_CARD."""
    eval_dir = models_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(eval_dir / "metrics_comparison.csv")
    _write_model_card(
        df, data.split_meta["dataset_hash"], models_dir / "MODEL_CARD.md", ctx
    )


def main() -> None:
    """Runs the comparative evaluation and writes artifacts."""
    settings = get_settings()
    config = load_config()
    set_seed(settings.random_seed)
    k = config["evaluation"]["k"]
    cap = config["evaluation"]["sample_size"]
    models_dir = settings.models_dir
    data = load_processed(settings.processed_data_dir)
    gt = data.test_ground_truth
    eval_users = _sample_users(list(gt.keys()), cap, settings.random_seed)
    n_eval = len(eval_users)
    rows = _collect_rows(models_dir, data.interactions, eval_users, gt, k)
    df = pd.DataFrame(rows).T.rename_axis("model")
    best = select_promoted_model(df)
    _save_artifacts(df, data, models_dir, _card_context(settings, config, best, n_eval))
    _register(df, best, data.split_meta["dataset_hash"], settings)
    logger.info("evaluation_done", best_model=best, eval_users=n_eval)


_MODEL_CARD_TEMPLATE = """# Model Card - Sistema de Recomendacao Instacart

Comparacao de 5 modelos (top-{k}) no split de teste interno. Dataset hash: `{hash}...`.

Todos os modelos foram avaliados sobre a **mesma** populacao: {n_eval} usuarios do
split de teste (amostra uniforme com seed 42 quando o teste excede o teto
`evaluation.sample_size`). Isso garante que as metricas e o `coverage_at_k` sejam
diretamente comparaveis entre modelos.

## Performance (split de teste)

{table}

## Decisao de Promocao

- NCF supera todos os baselines em recall@k: {beats_recall}
- NCF supera todos os baselines em ndcg@k: {beats_ndcg}
- Modelo com melhor recall@k: **{best}**
- `{registered_name}` (pyfunc servindo **{best}**) promovido via stages
  Staging -> Production e alias **@production** no MLflow Model Registry.

## Limitacoes

- **Avaliacao offline**: metricas medidas em split historico; sem teste online/A-B,
  entao exposicao, novidade e satisfacao real nao sao capturadas.
- **Feedback implicito**: o ground truth sao itens comprados no periodo de teste;
  ausencia de compra nao significa irrelevancia, o que subestima itens nunca expostos.
- **Cold-start**: usuarios ou itens sem historico caem no fallback de popularidade,
  sem personalizacao.
- **NCF** (modelo principal em PyTorch) {ncf_verdict}; o trade-off e documentado, mas
  o melhor modelo por recall@k e promovido conforme o edital.
- **Catalogo restrito** aos {top_n} produtos mais comprados: itens de cauda longa
  ficam fora da avaliacao e do serving.
- **Variancia amostral**: quando o teste excede o teto, as metricas vem de uma amostra
  de {n_eval} usuarios e carregam incerteza estatistica.

## Possiveis Vieses

- **Vies de popularidade**: o promovido (**{best}**) tem coverage@{k} = {best_cov};
  cobertura baixa concentra recomendacoes em itens ja populares (efeito "rico fica
  mais rico" / filter bubble) e reduz diversidade e descoberta.
- **Vies de recompra** do Instacart: usuarios reabastecem os mesmos produtos,
  favorecendo modelos que exploram o historico (item-based CF) e penalizando novidade.
- **Vies de selecao**: o recorte de catalogo top-{top_n} e o split por usuario favorecem
  itens e usuarios mais ativos; quem tem pouco historico e sub-representado.
- **Dados sinteticos**: quando usados no lugar do Kaggle real, nao reproduzem o vies de
  recompra e os resultados nao transferem diretamente.

Gerado automaticamente por `src/pipeline/evaluate.py`.
"""


if __name__ == "__main__":
    main()
