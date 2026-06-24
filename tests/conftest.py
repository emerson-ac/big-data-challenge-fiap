"""Fixtures compartilhadas pelos testes do módulo de predição."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from src.models.inference import RecommendationEngine


@pytest.fixture
def toy_similarity() -> sp.csr_matrix:
    """Similaridade item-item 4x4: item0~item1 e item2~item3."""
    dense = np.array(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.2, 1.0],
        ],
        dtype=np.float32,
    )
    return sp.csr_matrix(dense)


@pytest.fixture
def toy_interactions() -> sp.csr_matrix:
    """Histórico: usuário 0 comprou o item 0; usuário 1 não comprou nada."""
    dense = np.array([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
    return sp.csr_matrix(dense)


@pytest.fixture
def toy_vocab() -> dict:
    """Vocabulário com 2 usuários conhecidos e catálogo de 4 produtos."""
    return {
        "user_id_to_idx": {10: 0, 20: 1},
        "idx_to_product_id": np.array([100, 200, 300, 400]),
    }


@pytest.fixture
def toy_popularity_ranking() -> np.ndarray:
    """Ranking de popularidade global (item_idx do mais ao menos popular)."""
    return np.array([3, 2, 1, 0])


@pytest.fixture
def model_artifacts(
    tmp_path: Path,
    toy_similarity: sp.csr_matrix,
    toy_interactions: sp.csr_matrix,
    toy_vocab: dict,
    toy_popularity_ranking: np.ndarray,
) -> dict[str, Path]:
    """Persiste os artefatos sintéticos em disco e retorna seus caminhos."""
    similarity_path = tmp_path / "item_similarity.npz"
    interactions_path = tmp_path / "interactions_prior.npz"
    popularity_path = tmp_path / "ranking.pkl"
    vocab_path = tmp_path / "vocabularies.pkl"

    sp.save_npz(similarity_path, toy_similarity)
    sp.save_npz(interactions_path, toy_interactions)
    with open(popularity_path, "wb") as f:
        pickle.dump(toy_popularity_ranking, f)
    with open(vocab_path, "wb") as f:
        pickle.dump(toy_vocab, f)

    return {
        "similarity_path": similarity_path,
        "interactions_path": interactions_path,
        "popularity_path": popularity_path,
        "vocab_path": vocab_path,
    }


@pytest.fixture
def engine(model_artifacts: dict) -> RecommendationEngine:
    """RecommendationEngine real, carregado a partir dos artefatos sintéticos."""
    return RecommendationEngine(
        similarity_path=model_artifacts["similarity_path"],
        interactions_path=model_artifacts["interactions_path"],
        popularity_path=model_artifacts["popularity_path"],
        vocab_path=model_artifacts["vocab_path"],
    )
