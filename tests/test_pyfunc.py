"""Tests for the pyfunc wrapper registered in the MLflow Model Registry.

This is the object actually promoted to Production, so a break here ships a
broken model even when every training metric looks fine.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.serving.pyfunc import RecommenderPyfunc, build_artifacts


class _FakeContext:
    """Minimal stand-in for ``PythonModelContext`` (only exposes artifacts)."""

    def __init__(self, artifacts: dict[str, str]) -> None:
        self.artifacts = artifacts


@pytest.fixture
def pyfunc_artifacts(tmp_path: Path) -> dict[str, str]:
    """Writes the artifacts consumed by the popularity and item-CF scorers."""
    ranking_path = tmp_path / "ranking.pkl"
    similarity_path = tmp_path / "item_similarity.npz"
    interactions_path = tmp_path / "interactions_prior.npz"
    factors_dir = tmp_path

    with open(ranking_path, "wb") as f:
        pickle.dump(np.array([3, 2, 1, 0]), f)
    sp.save_npz(
        similarity_path,
        sp.csr_matrix(np.eye(4, dtype=np.float32) + np.eye(4, k=1, dtype=np.float32)),
    )
    sp.save_npz(
        interactions_path,
        sp.csr_matrix(np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.int8)),
    )
    np.save(factors_dir / "user_factors.npy", np.array([[1.0, 0.0], [0.0, 1.0]]))
    np.save(factors_dir / "item_factors.npy", np.eye(2))

    return {
        "ranking": str(ranking_path),
        "similarity": str(similarity_path),
        "interactions": str(interactions_path),
        "user_factors": str(factors_dir / "user_factors.npy"),
        "item_factors": str(factors_dir / "item_factors.npy"),
    }


def test_build_artifacts_covers_every_model_type() -> None:
    """The 5 comparable models must all be registrable as a pyfunc."""
    models_dir, processed_dir = Path("models"), Path("data/processed")

    for model_type in (
        "popularity",
        "item_based_cf",
        "user_based_cf",
        "matrix_factorization",
        "ncf",
    ):
        assert build_artifacts(model_type, models_dir, processed_dir)


def test_build_artifacts_rejects_unknown_model_type() -> None:
    """An unknown model type fails at registration, not at inference."""
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_artifacts("nope", Path("models"), Path("data/processed"))


def test_load_context_rejects_unknown_model_type(
    pyfunc_artifacts: dict[str, str],
) -> None:
    """A wrapper built with a bad model type fails when the scorer is created."""
    model = RecommenderPyfunc("nope")

    with pytest.raises(ValueError, match="Unknown model_type"):
        model.load_context(_FakeContext(pyfunc_artifacts))


def test_popularity_scorer_ranks_by_global_popularity(
    pyfunc_artifacts: dict[str, str],
) -> None:
    """Ranking [3, 2, 1, 0] must score item 3 highest for any user."""
    model = RecommenderPyfunc("popularity")
    model.load_context(_FakeContext(pyfunc_artifacts))

    scores = model.score_user(0)

    assert int(np.argmax(scores)) == 3


def test_popularity_scorer_ignores_the_user(pyfunc_artifacts: dict[str, str]) -> None:
    """The baseline is user-independent, so both users get the same vector."""
    model = RecommenderPyfunc("popularity")
    model.load_context(_FakeContext(pyfunc_artifacts))

    assert np.array_equal(model.score_user(0), model.score_user(1))


def test_item_cf_scorer_uses_the_user_history(
    pyfunc_artifacts: dict[str, str],
) -> None:
    """User 0 bought item 0, so the similar item 1 must score above item 3."""
    model = RecommenderPyfunc("item_based_cf")
    model.load_context(_FakeContext(pyfunc_artifacts))

    scores = model.score_user(0)

    assert scores[1] > scores[3]


def test_item_cf_scorer_returns_one_score_per_item(
    pyfunc_artifacts: dict[str, str],
) -> None:
    """The score vector must be dense and catalog-sized (4 items here)."""
    model = RecommenderPyfunc("item_based_cf")
    model.load_context(_FakeContext(pyfunc_artifacts))

    assert model.score_user(0).shape == (4,)


def test_mf_scorer_multiplies_user_and_item_factors(
    pyfunc_artifacts: dict[str, str],
) -> None:
    """With identity item factors, user 0's factors are returned as scores."""
    model = RecommenderPyfunc("matrix_factorization")
    model.load_context(_FakeContext(pyfunc_artifacts))

    assert np.array_equal(model.score_user(0), np.array([1.0, 0.0]))


def test_predict_returns_top_k_per_input_row(
    pyfunc_artifacts: dict[str, str],
) -> None:
    """Each row of the input frame yields its own top-k list."""
    model = RecommenderPyfunc("popularity")
    context = _FakeContext(pyfunc_artifacts)
    model.load_context(context)

    result = model.predict(context, pd.DataFrame({"user_idx": [0, 1], "k": [2, 3]}))

    assert result == [[3, 2], [3, 2, 1]]


def test_predict_output_is_json_serializable(
    pyfunc_artifacts: dict[str, str],
) -> None:
    """The API serializes the response, so numpy ints must not leak through."""
    model = RecommenderPyfunc("popularity")
    context = _FakeContext(pyfunc_artifacts)
    model.load_context(context)

    result = model.predict(context, pd.DataFrame({"user_idx": [0], "k": [2]}))

    assert json.dumps(result) == "[[3, 2]]"


@pytest.fixture
def user_cf_artifacts(tmp_path: Path, pyfunc_artifacts: dict[str, str]) -> dict:
    """Adds a fitted KNN bundle to the shared artifacts."""
    from sklearn.neighbors import NearestNeighbors

    interactions = sp.load_npz(pyfunc_artifacts["interactions"])
    pool_indices = np.array([0, 1])
    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=1)
    knn.fit(interactions[pool_indices])
    knn_path = tmp_path / "knn_model.pkl"
    with open(knn_path, "wb") as f:
        pickle.dump({"model": knn, "pool_indices": pool_indices}, f)
    return {**pyfunc_artifacts, "knn": str(knn_path)}


def test_user_cf_scorer_uses_the_neighbour_history(user_cf_artifacts: dict) -> None:
    """User 1 bought item 2; its nearest neighbour's history scores item 2 top."""
    model = RecommenderPyfunc("user_based_cf")
    model.load_context(_FakeContext(user_cf_artifacts))

    scores = model.score_user(1)

    assert int(np.argmax(scores)) == 2


@pytest.fixture
def ncf_artifacts(tmp_path: Path) -> dict[str, str]:
    """Persists a tiny trained-shape NCF plus the params and vocabulary."""
    import torch

    from src.models.ncf import NeuralCollaborativeFiltering

    params = {"embedding_dim": 4, "hidden_dims": [8]}
    net = NeuralCollaborativeFiltering(2, 4, params["embedding_dim"], (8,))
    model_path = tmp_path / "model.pt"
    torch.save(net.state_dict(), model_path)

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"params": params}), encoding="utf-8")

    vocab_path = tmp_path / "vocabularies.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(
            {"user_id_to_idx": {10: 0, 20: 1}, "idx_to_product_id": np.arange(4)}, f
        )

    return {
        "model": str(model_path),
        "metrics": str(metrics_path),
        "vocab": str(vocab_path),
    }


def test_ncf_scorer_returns_one_probability_per_item(
    ncf_artifacts: dict[str, str],
) -> None:
    """The main model must score the whole catalog in the sigmoid range."""
    model = RecommenderPyfunc("ncf")
    model.load_context(_FakeContext(ncf_artifacts))

    scores = model.score_user(0)

    assert scores.shape == (4,)
    assert ((scores > 0.0) & (scores < 1.0)).all()


def test_ncf_scorer_is_deterministic(ncf_artifacts: dict[str, str]) -> None:
    """Dropout must be off at inference: two calls give identical scores."""
    model = RecommenderPyfunc("ncf")
    model.load_context(_FakeContext(ncf_artifacts))

    assert np.array_equal(model.score_user(0), model.score_user(0))
