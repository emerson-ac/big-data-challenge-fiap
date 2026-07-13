"""Treino do baseline de popularidade (porta do notebook 03)."""

from pathlib import Path

import numpy as np

from src.evaluation.metrics import evaluate_recommendations
from src.models.training.data import ProcessedData, save_metrics, save_pickle


def _recs_for(ranking: np.ndarray, ground_truth: dict, k: int) -> dict:
    """Recomenda o mesmo top-k global para todos os usuários do ground truth."""
    top = ranking[:k].tolist()
    return {user_idx: top for user_idx in ground_truth}


def train(data: ProcessedData, k: int, out_dir: Path) -> dict:
    """Treina e avalia o recomendador de popularidade global.

    Args:
        data: Artefatos processados (interações + ground truths).
        k: Tamanho do top-k avaliado.
        out_dir: Diretório de saída do modelo.

    Returns:
        Payload de métricas ({k, validation, test}).
    """
    frequency = np.asarray(data.interactions.sum(axis=0)).flatten()
    ranking = np.argsort(-frequency)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(out_dir / "ranking.pkl", ranking)

    val = evaluate_recommendations(
        _recs_for(ranking, data.val_ground_truth, k), data.val_ground_truth, k
    )
    test = evaluate_recommendations(
        _recs_for(ranking, data.test_ground_truth, k), data.test_ground_truth, k
    )
    payload = {"k": k, "validation": val, "test": test}
    save_metrics(out_dir, payload)
    return payload
