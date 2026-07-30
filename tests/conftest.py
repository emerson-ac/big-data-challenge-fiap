"""Shared fixtures for the inference module tests."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from src.models.inference import RecommendationEngine


@pytest.fixture
def toy_similarity() -> sp.csr_matrix:
    """4x4 item-item similarity: item0~item1 and item2~item3."""
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
    """History: user 0 bought item 0; user 1 bought nothing."""
    dense = np.array([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
    return sp.csr_matrix(dense)


@pytest.fixture
def toy_vocab() -> dict:
    """Vocabulary with 2 known users and a catalog of 4 products."""
    return {
        "user_id_to_idx": {10: 0, 20: 1},
        "idx_to_product_id": np.array([100, 200, 300, 400]),
    }


@pytest.fixture
def toy_popularity_ranking() -> np.ndarray:
    """Global popularity ranking (item_idx from most to least popular)."""
    return np.array([3, 2, 1, 0])


@pytest.fixture
def model_artifacts(
    tmp_path: Path,
    toy_similarity: sp.csr_matrix,
    toy_interactions: sp.csr_matrix,
    toy_vocab: dict,
    toy_popularity_ranking: np.ndarray,
) -> dict[str, Path]:
    """Persists the synthetic artifacts to disk and returns their paths."""
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
    """RecommendationEngine real, carregado a partir dos artefatos sintéticos.

    O ``model_type`` é fixado para o teste não depender de qual modelo o
    pipeline promoveu no ambiente onde a suíte roda.
    """
    return RecommendationEngine(
        model_type="item_based_cf",
        similarity_path=model_artifacts["similarity_path"],
        interactions_path=model_artifacts["interactions_path"],
        popularity_path=model_artifacts["popularity_path"],
        vocab_path=model_artifacts["vocab_path"],
    )
