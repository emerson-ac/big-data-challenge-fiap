"""Carregamento dos artefatos processados e avaliação compartilhada no treino."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp

from src.evaluation.metrics import evaluate_recommendations, pairs_to_ground_truth
from src.evaluation.ranking import recommendations_from_score_matrix


@dataclass
class ProcessedData:
    """Artefatos de ``data/processed`` consumidos pelos treinos.

    Attributes:
        interactions: Matriz esparsa (n_users, n_items) de histórico prior.
        val_ground_truth: Mapa user_idx -> itens relevantes de validação.
        test_ground_truth: Mapa user_idx -> itens relevantes de teste.
        split_meta: Metadados do split (inclui dataset_hash).
    """

    interactions: sp.csr_matrix
    val_ground_truth: dict[int, set[int]]
    test_ground_truth: dict[int, set[int]]
    split_meta: dict


def load_processed(processed_dir: Path) -> ProcessedData:
    """Carrega matriz de interações, ground truths e metadados do split.

    Args:
        processed_dir: Diretório ``data/processed``.

    Returns:
        Instância ProcessedData pronta para treino/avaliação.
    """
    import pandas as pd

    interactions = sp.load_npz(processed_dir / "interactions_prior.npz")
    val_gt = pairs_to_ground_truth(pd.read_pickle(processed_dir / "val_pairs.pkl"))
    test_gt = pairs_to_ground_truth(pd.read_pickle(processed_dir / "test_pairs.pkl"))
    with open(processed_dir / "split_meta.json", encoding="utf-8") as f:
        split_meta = json.load(f)
    return ProcessedData(interactions, val_gt, test_gt, split_meta)


def evaluate_score_fn(
    users: list[int],
    score_fn: Callable[[list[int]], np.ndarray],
    ground_truth: dict[int, set[int]],
    k: int,
) -> dict[str, float]:
    """Avalia um modelo dado um callable que produz a matriz de scores.

    Args:
        users: user_idx a avaliar.
        score_fn: Função que recebe os users e devolve a matriz de scores.
        ground_truth: Mapa user_idx -> itens relevantes.
        k: Tamanho do top-k.

    Returns:
        Dicionário com as 4 métricas oficiais.
    """
    scores = score_fn(users)
    recs = recommendations_from_score_matrix(users, scores, k)
    return evaluate_recommendations(recs, ground_truth, k)


def save_metrics(out_dir: Path, payload: dict) -> None:
    """Salva o metrics.json de um modelo.

    Args:
        out_dir: Diretório de saída do modelo.
        payload: Dicionário de métricas a serializar.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_pickle(path: Path, obj: object) -> None:
    """Serializa um objeto arbitrário em pickle.

    Args:
        path: Caminho de destino.
        obj: Objeto a persistir.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)
