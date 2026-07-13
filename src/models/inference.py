"""Recommendation prediction: entrypoint used by the service/API."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from src.config import get_settings
from src.evaluation.ranking import top_k_from_scores
from src.models.model_loader import ModelFactory

logger = structlog.get_logger()

DEFAULT_MODEL_TYPE = "item_based_cf"
DEFAULT_SIMILARITY_PATH = Path("models/item_based_cf/item_similarity.npz")
DEFAULT_INTERACTIONS_PATH = Path("data/processed/interactions_prior.npz")
DEFAULT_POPULARITY_PATH = Path("models/baseline_popularity/ranking.pkl")
DEFAULT_VOCAB_PATH = Path("data/processed/vocabularies.pkl")


@dataclass(frozen=True)
class Recommendation:
    """A recommended item with its position and score.

    Attributes:
        product_id: External product identifier.
        score: Affinity score computed by the model.
        rank: Position in the ranking (1-based).
    """

    product_id: int
    score: float
    rank: int


def load_vocabularies(vocab_path: Path = DEFAULT_VOCAB_PATH) -> dict[str, Any]:
    """Loads the vocabulary map (user/product <-> index).

    Args:
        vocab_path: Path to the vocabulary pickle.

    Returns:
        Dictionary with user_id_to_idx, idx_to_product_id, etc.
    """
    with open(vocab_path, "rb") as f:
        return pickle.load(f)


def _build_primary_model(
    source: str, model_type: str, similarity_path: Path, interactions_path: Path
) -> Any:
    """Instantiates the primary model per the source (registry or local).

    Args:
        source: "registry" (MLflow) or "local" (disk artifacts).
        model_type: Local model name registered in ModelFactory.
        similarity_path: Path to item-item similarity (local mode).
        interactions_path: Path to purchase history (local mode).

    Returns:
        Recommender with a ``score_user`` method.
    """
    if source == "registry":
        settings = get_settings()
        return ModelFactory.create(
            "item_based_cf_registry",
            name=settings.registered_model_name,
            alias=settings.model_alias,
        )
    return ModelFactory.create(
        model_type,
        similarity_path=similarity_path,
        interactions_path=interactions_path,
    )


class RecommendationEngine:
    """Orchestrates the Production model and the popularity fallback.

    Args:
        model_type: Name of the model registered in ModelFactory.
        similarity_path: Path to the item-item similarity artifact.
        interactions_path: Path to the purchase history (sparse).
        popularity_path: Path to the popularity ranking (cold-start fallback).
        vocab_path: Path to the user/product vocabularies.
    """

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_TYPE,
        similarity_path: Path = DEFAULT_SIMILARITY_PATH,
        interactions_path: Path = DEFAULT_INTERACTIONS_PATH,
        popularity_path: Path = DEFAULT_POPULARITY_PATH,
        vocab_path: Path = DEFAULT_VOCAB_PATH,
        model_source: str | None = None,
    ) -> None:
        source = model_source or get_settings().model_source
        self._model = _build_primary_model(
            source, model_type, similarity_path, interactions_path
        )
        self._fallback = ModelFactory.create("popularity", ranking_path=popularity_path)
        self._vocab = load_vocabularies(vocab_path)
        logger.info("recommendation_engine_loaded", model_source=source)

    def is_known_user(self, user_id: int) -> bool:
        """Verifica se o usuário tem histórico conhecido pelo modelo treinado.

        Args:
            user_id: ID externo do usuário.

        Returns:
            True se o usuário existe no vocabulário de treino, False se é cold-start.
        """
        return user_id in self._vocab["user_id_to_idx"]

    @property
    def catalog_size(self) -> int:
        """Número de produtos no catálogo conhecido pelo modelo."""
        return len(self._vocab["idx_to_product_id"])

    def recommend(self, user_id: int, k: int = 10) -> list[Recommendation]:
        """Generates the top-k recommendations for an external user.

        Args:
            user_id: External user id (e.g. Instacart user_id).
            k: Number of recommendations desired.

        Returns:
            List of Recommendation ordered by descending score.
        """
        user_idx = self._vocab["user_id_to_idx"].get(user_id)
        if user_idx is None:
            logger.info("cold_start_user", user_id=user_id)
            return self._recommend_popular(k)
        scores = self._model.score_user(user_idx)
        return self._build_recommendations(scores, k)

    def _recommend_popular(self, k: int) -> list[Recommendation]:
        """Builds recommendations from the popularity ranking."""
        item_indices = self._fallback.top_k(k)
        idx_to_product_id = self._vocab["idx_to_product_id"]
        return [
            Recommendation(
                product_id=int(idx_to_product_id[item_idx]), score=0.0, rank=rank
            )
            for rank, item_idx in enumerate(item_indices, 1)
        ]

    def _build_recommendations(
        self, scores: np.ndarray, k: int
    ) -> list[Recommendation]:
        """Turns a score vector into a ranked list of Recommendation."""
        top_items = top_k_from_scores(scores, k)
        idx_to_product_id = self._vocab["idx_to_product_id"]
        return [
            Recommendation(
                product_id=int(idx_to_product_id[item_idx]),
                score=float(scores[item_idx]),
                rank=rank,
            )
            for rank, item_idx in enumerate(top_items, 1)
        ]
