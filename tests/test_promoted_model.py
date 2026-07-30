"""Tests that the API serves the model the evaluation stage promoted.

The failure this guards against is silent: the pipeline promotes one model to
Production while the API keeps serving another, and every metric still looks
fine because nothing compares the two.
"""

import json
from pathlib import Path

import pytest

from src.models.inference import (
    DEFAULT_MODEL_TYPE,
    RecommendationEngine,
    read_promoted_model_type,
)


def _write_marker(path: Path, model_type: str) -> Path:
    """Writes a promoted-model marker like the evaluate stage does."""
    path.write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return path


def test_reads_the_promoted_model_type(tmp_path: Path) -> None:
    """The marker written by the evaluate stage is what gets served."""
    marker = _write_marker(tmp_path / "promoted_model.json", "matrix_factorization")

    assert read_promoted_model_type(marker) == "matrix_factorization"


def test_falls_back_when_marker_is_absent(tmp_path: Path) -> None:
    """A fresh clone that never ran the pipeline still starts the API."""
    assert read_promoted_model_type(tmp_path / "missing.json") == DEFAULT_MODEL_TYPE


def test_falls_back_when_marker_is_malformed(tmp_path: Path) -> None:
    """Corrupt JSON must not crash startup."""
    marker = tmp_path / "promoted_model.json"
    marker.write_text("{not json", encoding="utf-8")

    assert read_promoted_model_type(marker) == DEFAULT_MODEL_TYPE


def test_falls_back_when_marker_lacks_the_key(tmp_path: Path) -> None:
    """A marker from an older format is treated as absent."""
    marker = tmp_path / "promoted_model.json"
    marker.write_text(json.dumps({"recall_at_k": 0.5}), encoding="utf-8")

    assert read_promoted_model_type(marker) == DEFAULT_MODEL_TYPE


def test_engine_reports_the_model_type_it_loaded(model_artifacts: dict) -> None:
    """model_type reflects what is served, not a configured default."""
    engine = RecommendationEngine(
        model_type="item_based_cf",
        similarity_path=model_artifacts["similarity_path"],
        interactions_path=model_artifacts["interactions_path"],
        popularity_path=model_artifacts["popularity_path"],
        vocab_path=model_artifacts["vocab_path"],
    )

    assert engine.model_type == "item_based_cf"


def _models_tree(root: Path, ranking_source: Path) -> Path:
    """Builds the ``models/`` layout that build_artifacts expects."""
    target = root / "baseline_popularity"
    target.mkdir(parents=True, exist_ok=True)
    target.joinpath("ranking.pkl").write_bytes(ranking_source.read_bytes())
    return root


def test_engine_follows_the_marker_when_type_is_not_forced(
    tmp_path: Path, model_artifacts: dict
) -> None:
    """With no explicit type, the engine adopts the promoted one."""
    marker = _write_marker(tmp_path / "promoted_model.json", "popularity")
    models_dir = _models_tree(tmp_path / "models", model_artifacts["popularity_path"])

    engine = RecommendationEngine(
        similarity_path=model_artifacts["similarity_path"],
        interactions_path=model_artifacts["interactions_path"],
        popularity_path=model_artifacts["popularity_path"],
        vocab_path=model_artifacts["vocab_path"],
        promoted_marker=marker,
        models_dir=models_dir,
        processed_dir=tmp_path,
    )

    assert engine.model_type == "popularity"


def test_engine_serves_the_promoted_model_not_the_default(
    tmp_path: Path, model_artifacts: dict
) -> None:
    """Promoted popularity must rank by global popularity, not by item-CF.

    The toy ranking is [3, 2, 1, 0] while item-CF would put item 1 on top for
    user 10 (who bought item 0), so the top recommendation tells them apart.
    """
    marker = _write_marker(tmp_path / "promoted_model.json", "popularity")
    models_dir = _models_tree(tmp_path / "models", model_artifacts["popularity_path"])

    engine = RecommendationEngine(
        similarity_path=model_artifacts["similarity_path"],
        interactions_path=model_artifacts["interactions_path"],
        popularity_path=model_artifacts["popularity_path"],
        vocab_path=model_artifacts["vocab_path"],
        promoted_marker=marker,
        models_dir=models_dir,
        processed_dir=tmp_path,
    )

    top = engine.recommend(user_id=10, k=1)[0]

    assert top.product_id == 400


def test_marker_written_by_evaluate_is_readable_by_the_api(tmp_path: Path) -> None:
    """Closes the loop: the producer and the consumer agree on the format.

    If the evaluate stage ever changes the marker schema, this fails instead of
    the API silently falling back to the default model.
    """
    import pandas as pd

    from src.config import Settings
    from src.pipeline.evaluate import _write_promoted_marker

    df = pd.DataFrame({"recall_at_k": [0.42]}, index=["ncf"]).rename_axis("model")

    class _Data:
        split_meta = {"dataset_hash": "abc123"}

    marker = tmp_path / "promoted_model.json"
    _write_promoted_marker(df, "ncf", _Data(), marker, Settings())

    assert read_promoted_model_type(marker) == "ncf"


def test_unknown_promoted_type_fails_loudly(
    tmp_path: Path, model_artifacts: dict
) -> None:
    """A marker naming a model the serving layer cannot build must not start."""
    marker = _write_marker(tmp_path / "promoted_model.json", "nope")

    with pytest.raises(ValueError, match="Unknown model_type"):
        RecommendationEngine(
            similarity_path=model_artifacts["similarity_path"],
            interactions_path=model_artifacts["interactions_path"],
            popularity_path=model_artifacts["popularity_path"],
            vocab_path=model_artifacts["vocab_path"],
            promoted_marker=marker,
        )
