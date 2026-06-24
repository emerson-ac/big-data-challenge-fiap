"""Testes da camada de predição (RecommendationEngine)."""

from src.models.inference import (
    Recommendation,
    RecommendationEngine,
    load_vocabularies,
)


def _build_engine(model_artifacts: dict) -> RecommendationEngine:
    """Instancia o RecommendationEngine com artefatos sintéticos de teste."""
    return RecommendationEngine(
        similarity_path=model_artifacts["similarity_path"],
        interactions_path=model_artifacts["interactions_path"],
        popularity_path=model_artifacts["popularity_path"],
        vocab_path=model_artifacts["vocab_path"],
    )


def test_load_vocabularies_returns_expected_keys(model_artifacts: dict) -> None:
    """load_vocabularies restaura o dicionário de mapas persistido."""
    vocab = load_vocabularies(model_artifacts["vocab_path"])

    assert vocab["user_id_to_idx"] == {10: 0, 20: 1}


def test_recommend_known_user_uses_model_scores(model_artifacts: dict) -> None:
    """Usuário conhecido recebe recomendações ranqueadas pelo score do modelo."""
    engine = _build_engine(model_artifacts)

    recommendations = engine.recommend(user_id=10, k=2)

    assert recommendations == [
        Recommendation(product_id=100, score=1.0, rank=1),
        Recommendation(product_id=200, score=0.5, rank=2),
    ]


def test_recommend_unknown_user_falls_back_to_popularity(
    model_artifacts: dict,
) -> None:
    """Usuário desconhecido (cold-start) recebe o ranking de popularidade."""
    engine = _build_engine(model_artifacts)

    recommendations = engine.recommend(user_id=999, k=2)

    assert recommendations == [
        Recommendation(product_id=400, score=0.0, rank=1),
        Recommendation(product_id=300, score=0.0, rank=2),
    ]


def test_recommend_respects_k(model_artifacts: dict) -> None:
    """O número de recomendações retornadas respeita o parâmetro k."""
    engine = _build_engine(model_artifacts)

    recommendations = engine.recommend(user_id=10, k=1)

    assert len(recommendations) == 1
