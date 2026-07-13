"""Treino do Matrix Factorization implícito (porta do notebook 06, TruncatedSVD)."""

from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

from src.models.training.data import ProcessedData, evaluate_score_fn, save_metrics


def _fit(interactions, n_components: int, seed: int) -> tuple:
    """Ajusta TruncatedSVD e devolve (user_factors, item_factors, var_explicada)."""
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    user_factors = svd.fit_transform(interactions.astype("float32"))
    item_factors = svd.components_.T
    return user_factors, item_factors, float(svd.explained_variance_ratio_.sum())


def _score_fn(user_factors: np.ndarray, item_factors: np.ndarray):
    """Devolve um callable users -> matriz densa de scores (produto de fatores)."""
    return lambda users: user_factors[users] @ item_factors.T


def _clamp(choices: list[int], n_items: int) -> list[int]:
    """Limita cada n_components a no máximo n_items-1 (restrição do TruncatedSVD)."""
    return sorted({min(c, n_items - 1) for c in choices if c > 0})


def train(data: ProcessedData, k: int, cfg: dict, out_dir: Path, seed: int) -> dict:
    """Treina o MF com random search sobre n_components.

    Args:
        data: Artefatos processados.
        k: Tamanho do top-k avaliado.
        cfg: Bloco ``matrix_factorization`` da configuração.
        out_dir: Diretório de saída do modelo.
        seed: Semente de aleatoriedade.

    Returns:
        Payload de métricas com search_results e explained_variance_ratio.
    """
    n_items = data.interactions.shape[1]
    val_users = list(data.val_ground_truth.keys())
    search, best = _run_search(data, val_users, k, cfg, n_items, seed)

    user_factors, item_factors, variance = _fit(data.interactions, best, seed)
    score_fn = _score_fn(user_factors, item_factors)
    val = evaluate_score_fn(val_users, score_fn, data.val_ground_truth, k)
    test = evaluate_score_fn(
        list(data.test_ground_truth), score_fn, data.test_ground_truth, k
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "user_factors.npy", user_factors)
    np.save(out_dir / "item_factors.npy", item_factors)
    payload = {
        "k": k,
        "n_components": best,
        "explained_variance_ratio": variance,
        "search_results": search,
        "validation": val,
        "test": test,
    }
    save_metrics(out_dir, payload)
    return payload


def _run_search(data, val_users, k, cfg, n_items, seed) -> tuple[list[dict], int]:
    """Random search sobre n_components_choices; retorna (resultados, melhor)."""
    rng = np.random.default_rng(seed)
    choices = _clamp(cfg["search"]["n_components_choices"], n_items)
    trials = rng.choice(
        choices, size=min(cfg["search"]["n_trials"], len(choices)), replace=False
    )
    results = []
    for n_comp in trials:
        user_factors, item_factors, _ = _fit(data.interactions, int(n_comp), seed)
        metrics = evaluate_score_fn(
            val_users, _score_fn(user_factors, item_factors), data.val_ground_truth, k
        )
        results.append({"n_components": int(n_comp), **metrics})
    best = max(results, key=lambda r: r["recall_at_k"])["n_components"]
    return results, best
