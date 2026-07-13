"""Treino do user-based CF via KNN (porta do notebook 05, avaliação amostrada)."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from src.evaluation.metrics import evaluate_recommendations
from src.evaluation.ranking import recommendations_from_score_matrix
from src.models.training.data import ProcessedData, save_metrics, save_pickle


def build_recommendations(
    users: list[int],
    knn,
    pool_matrix: sp.csr_matrix,
    interactions,
    n_neighbors: int,
    k: int,
) -> dict:
    """Gera recomendações ponderadas pelos vizinhos mais próximos (cosine).

    Args:
        users: user_idx a recomendar.
        knn: NearestNeighbors ajustado sobre o pool.
        pool_matrix: Interações dos usuários do pool.
        interactions: Matriz de interações completa.
        n_neighbors: Vizinhos considerados por usuário.
        k: Tamanho do top-k.

    Returns:
        Mapa user_idx -> lista de itens recomendados.
    """
    distances, neighbors = knn.kneighbors(interactions[users], n_neighbors=n_neighbors)
    similarities = 1.0 - distances
    rows = np.repeat(np.arange(len(users)), n_neighbors)
    weights = sp.csr_matrix(
        (similarities.flatten(), (rows, neighbors.flatten())),
        shape=(len(users), pool_matrix.shape[0]),
    )
    scores = weights.dot(pool_matrix).toarray()
    return recommendations_from_score_matrix(users, scores, k)


def _sample(rng, population: list[int], size: int) -> list[int]:
    """Amostra sem reposição até ``size`` elementos da população."""
    return rng.choice(
        population, size=min(size, len(population)), replace=False
    ).tolist()


def train(data: ProcessedData, k: int, cfg: dict, out_dir: Path, seed: int) -> dict:
    """Treina o user-based CF com random search sobre n_neighbors (amostrado).

    Args:
        data: Artefatos processados.
        k: Tamanho do top-k avaliado.
        cfg: Bloco ``user_based_cf`` da configuração.
        out_dir: Diretório de saída do modelo.
        seed: Semente de aleatoriedade.

    Returns:
        Payload de métricas com search_results.
    """
    rng = np.random.default_rng(seed)
    n_users = data.interactions.shape[0]
    pool = rng.choice(
        n_users, size=min(cfg["neighbor_pool_size"], n_users), replace=False
    )
    pool_matrix = data.interactions[pool]
    knn = NearestNeighbors(metric="cosine", algorithm="brute").fit(pool_matrix)

    sample_n = cfg["eval_sample_size"]
    val_sample = _sample(rng, list(data.val_ground_truth), sample_n)
    test_sample = _sample(rng, list(data.test_ground_truth), sample_n)
    search, best = _run_search(data, knn, pool_matrix, val_sample, k, cfg, len(pool))

    val = _eval(
        data.val_ground_truth, val_sample, knn, pool_matrix, data.interactions, best, k
    )
    test = _eval(
        data.test_ground_truth,
        test_sample,
        knn,
        pool_matrix,
        data.interactions,
        best,
        k,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(out_dir / "knn_model.pkl", {"model": knn, "pool_indices": pool})
    payload = {
        "k": k,
        "n_neighbors": best,
        "neighbor_pool_size": int(len(pool)),
        "eval_sample_size": sample_n,
        "search_results": search,
        "validation": val,
        "test": test,
    }
    save_metrics(out_dir, payload)
    return payload


def _eval(ground_truth, sample, knn, pool_matrix, interactions, n_neighbors, k) -> dict:
    """Avalia recomendações user-based numa amostra de usuários."""
    recs = build_recommendations(
        sample,
        knn,
        pool_matrix,
        interactions,
        min(n_neighbors, pool_matrix.shape[0]),
        k,
    )
    gt = {u: ground_truth[u] for u in sample}
    return evaluate_recommendations(recs, gt, k)


def _run_search(
    data, knn, pool_matrix, val_sample, k, cfg, pool_size
) -> tuple[list[dict], int]:
    """Random search sobre n_neighbors_choices na amostra de validação."""
    rng = np.random.default_rng(42)
    choices = [c for c in cfg["search"]["n_neighbors_choices"] if c <= pool_size]
    trials = rng.choice(
        choices, size=min(cfg["search"]["n_trials"], len(choices)), replace=False
    )
    results = []
    for n in trials:
        metrics = _eval(
            data.val_ground_truth,
            val_sample,
            knn,
            pool_matrix,
            data.interactions,
            int(n),
            k,
        )
        results.append({"n_neighbors": int(n), **metrics})
    best = max(results, key=lambda r: r["recall_at_k"])["n_neighbors"]
    return results, best
