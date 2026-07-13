"""Treino do item-based CF (porta do notebook 04, modelo Production)."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from src.models.training.data import ProcessedData, evaluate_score_fn, save_metrics


def truncate_similarity(similarity: np.ndarray, top_m: int) -> np.ndarray:
    """Mantém as top_m maiores similaridades por linha, zerando o resto.

    Args:
        similarity: Matriz densa (n_items, n_items) de similaridade.
        top_m: Número de vizinhos mantidos por item.

    Returns:
        Matriz densa truncada.
    """
    if top_m >= similarity.shape[1]:
        return similarity
    truncated = np.zeros_like(similarity)
    for i in range(similarity.shape[0]):
        idx = np.argpartition(-similarity[i], top_m)[:top_m]
        truncated[i, idx] = similarity[i, idx]
    return truncated


def _score_fn(interactions: sp.csr_matrix, similarity: np.ndarray):
    """Devolve um callable users -> matriz densa de scores."""
    return lambda users: interactions[users].dot(similarity)


def train(data: ProcessedData, k: int, cfg: dict, out_dir: Path) -> dict:
    """Treina o item-based CF com random search sobre o top_m.

    Args:
        data: Artefatos processados.
        k: Tamanho do top-k avaliado.
        cfg: Bloco ``item_based_cf`` da configuração.
        out_dir: Diretório de saída do modelo.

    Returns:
        Payload de métricas ({k, top_m, search_results, validation, test}).
    """
    similarity = cosine_similarity(data.interactions.T, dense_output=True).astype(
        "float32"
    )
    val_users = list(data.val_ground_truth.keys())
    search = _run_search(data, similarity, val_users, k, cfg)
    best_top_m = max(search, key=lambda r: r["recall_at_k"])["top_m"]

    final = truncate_similarity(similarity, best_top_m)
    score_fn = _score_fn(data.interactions, final)
    val = evaluate_score_fn(val_users, score_fn, data.val_ground_truth, k)
    test = evaluate_score_fn(
        list(data.test_ground_truth), score_fn, data.test_ground_truth, k
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "item_similarity.npz", sp.csr_matrix(final))
    payload = {
        "k": k,
        "top_m": best_top_m,
        "search_results": search,
        "validation": val,
        "test": test,
    }
    save_metrics(out_dir, payload)
    return payload


def _run_search(data, similarity, val_users, k, cfg) -> list[dict]:
    """Random search sobre top_m_choices, avaliando recall na validação."""
    rng = np.random.default_rng(42)
    choices = cfg["search"]["top_m_choices"]
    trials = rng.choice(
        choices, size=min(cfg["search"]["n_trials"], len(choices)), replace=False
    )
    results = []
    for top_m in trials:
        trunc = truncate_similarity(similarity, int(top_m))
        metrics = evaluate_score_fn(
            val_users, _score_fn(data.interactions, trunc), data.val_ground_truth, k
        )
        results.append({"top_m": int(top_m), **metrics})
    return results
