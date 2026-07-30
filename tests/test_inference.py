"""Tests for the prediction layer (RecommendationEngine)."""

from src.models.inference import (
    Recommendation,
    RecommendationEngine,
    load_vocabularies,
)


def _build_engine(model_artifacts: dict) -> RecommendationEngine:
    """Instantiates the RecommendationEngine with synthetic test artifacts."""
    return RecommendationEngine(
        similarity_path=model_artifacts["similarity_path"],
        interactions_path=model_artifacts["interactions_path"],
        popularity_path=model_artifacts["popularity_path"],
        vocab_path=model_artifacts["vocab_path"],
    )


def test_load_vocabularies_returns_expected_keys(model_artifacts: dict) -> None:
    """load_vocabularies restores the persisted mapping dictionary."""
    vocab = load_vocabularies(model_artifacts["vocab_path"])

    assert vocab["user_id_to_idx"] == {10: 0, 20: 1}


def test_recommend_known_user_uses_model_scores(model_artifacts: dict) -> None:
    """A known user receives recommendations ranked by the model score."""
    engine = _build_engine(model_artifacts)

    recommendations = engine.recommend(user_id=10, k=2)

    assert recommendations == [
        Recommendation(product_id=100, score=1.0, rank=1),
        Recommendation(product_id=200, score=0.5, rank=2),
    ]


def test_recommend_unknown_user_falls_back_to_popularity(
    model_artifacts: dict,
) -> None:
    """An unknown user (cold-start) receives the popularity ranking."""
    engine = _build_engine(model_artifacts)

    recommendations = engine.recommend(user_id=999, k=2)

    assert recommendations == [
        Recommendation(product_id=400, score=0.0, rank=1),
        Recommendation(product_id=300, score=0.0, rank=2),
    ]


def test_recommend_respects_k(model_artifacts: dict) -> None:
    """The number of returned recommendations respects the k parameter."""
    engine = _build_engine(model_artifacts)

    recommendations = engine.recommend(user_id=10, k=1)

    assert len(recommendations) == 1


def test_model_type_reports_local_default(model_artifacts: dict) -> None:
    """In local mode the engine reports the model it actually loaded."""
    engine = _build_engine(model_artifacts)

    assert engine.model_type == "item_based_cf"


def test_popularity_can_serve_as_primary_model(model_artifacts: dict) -> None:
    """A promoted popularity model is served (score_user), reported as such."""
    engine = RecommendationEngine(
        model_type="popularity",
        similarity_path=model_artifacts["similarity_path"],
        interactions_path=model_artifacts["interactions_path"],
        popularity_path=model_artifacts["popularity_path"],
        vocab_path=model_artifacts["vocab_path"],
    )

    recommendations = engine.recommend(user_id=10, k=2)

    assert engine.model_type == "popularity"
    assert [rec.product_id for rec in recommendations] == [400, 300]
