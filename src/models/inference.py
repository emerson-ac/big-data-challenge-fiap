"""Predição de recomendações: ponto de entrada usado pelo serviço/API."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

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
    """Item recomendado com posição e score.

    Attributes:
        product_id: Identificador externo do produto.
        score: Score de afinidade calculado pelo modelo.
        rank: Posição no ranking (1-based).
    """

    product_id: int
    score: float
    rank: int


def load_vocabularies(vocab_path: Path = DEFAULT_VOCAB_PATH) -> dict[str, Any]:
    """Carrega o dicionário de vocabulários (mapas usuário/produto <-> índice).

    Args:
        vocab_path: Caminho do pickle de vocabulários.

    Returns:
        Dicionário com as chaves user_id_to_idx, idx_to_product_id, etc.
    """
    with open(vocab_path, "rb") as f:
        return pickle.load(f)


class RecommendationEngine:
    """Orquestra o modelo Production e o fallback de popularidade.

    Args:
        model_type: Nome do modelo registrado na ModelFactory.
        similarity_path: Caminho do artefato de similaridade item-item.
        interactions_path: Caminho do histórico de compras (sparse).
        popularity_path: Caminho do ranking de popularidade (fallback cold-start).
        vocab_path: Caminho dos vocabulários de usuário/produto.
    """

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_TYPE,
        similarity_path: Path = DEFAULT_SIMILARITY_PATH,
        interactions_path: Path = DEFAULT_INTERACTIONS_PATH,
        popularity_path: Path = DEFAULT_POPULARITY_PATH,
        vocab_path: Path = DEFAULT_VOCAB_PATH,
    ) -> None:
        self._model = ModelFactory.create(
            model_type,
            similarity_path=similarity_path,
            interactions_path=interactions_path,
        )
        self._fallback = ModelFactory.create("popularity", ranking_path=popularity_path)
        self._vocab = load_vocabularies(vocab_path)
        logger.info("recommendation_engine_loaded", model_type=model_type)

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
        """Gera o top-k de recomendações para um usuário externo.

        Args:
            user_id: ID externo do usuário (ex.: user_id do Instacart).
            k: Quantidade de recomendações desejadas.

        Returns:
            Lista de Recommendation ordenada por score decrescente.
        """
        user_idx = self._vocab["user_id_to_idx"].get(user_id)
        if user_idx is None:
            logger.info("cold_start_user", user_id=user_id)
            return self._recommend_popular(k)
        scores = self._model.score_user(user_idx)
        return self._build_recommendations(scores, k)

    def _recommend_popular(self, k: int) -> list[Recommendation]:
        """Constrói recomendações a partir do ranking de popularidade."""
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
        """Converte um vetor de scores em uma lista ranqueada de Recommendation."""
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
