"""Tests for the evaluation stage: shared sampling and MODEL_CARD content."""

from pathlib import Path

import pandas as pd

from src.pipeline.evaluate import _ncf_verdict, _sample_users, _write_model_card


def test_sample_users_returns_all_when_under_cap() -> None:
    """When the split fits under the cap, every user is evaluated."""
    users = [5, 3, 9]

    assert _sample_users(users, cap=10, seed=42) == users


def test_sample_users_caps_size_and_is_reproducible() -> None:
    """Above the cap, the sample is bounded and deterministic for a fixed seed."""
    users = list(range(1000))

    first = _sample_users(users, cap=10, seed=42)
    second = _sample_users(users, cap=10, seed=42)

    assert first == second
    assert len(first) == 10
    assert set(first).issubset(set(users))


def test_sample_users_is_not_the_lowest_indices() -> None:
    """A seeded uniform sample avoids the first-N (low user_idx) bias."""
    users = list(range(1000))

    sample = _sample_users(users, cap=10, seed=42)

    assert sample != users[:10]


def test_ncf_verdict_reflects_baseline_comparison() -> None:
    """The NCF verdict text matches whether it beat the CF baselines."""
    assert "superou os baselines" in _ncf_verdict(True, True)
    assert "parcialmente" in _ncf_verdict(True, False)
    assert "nao superou" in _ncf_verdict(False, False)


def _toy_comparison() -> pd.DataFrame:
    """A 3-model comparison table with the columns the card needs."""
    return pd.DataFrame(
        {
            "recall_at_k": {"popularity": 0.50, "item_based_cf": 0.40, "ncf": 0.30},
            "ndcg_at_k": {"popularity": 0.50, "item_based_cf": 0.40, "ncf": 0.30},
            "coverage_at_k": {"popularity": 0.05, "item_based_cf": 0.20, "ncf": 0.10},
        }
    ).rename_axis("model")


def test_model_card_documents_performance_limitations_and_biases(
    tmp_path: Path,
) -> None:
    """The MODEL_CARD renders the edital sections with dynamic values."""
    out = tmp_path / "MODEL_CARD.md"
    ctx = {
        "registered_name": "recsys_recommender",
        "best": "popularity",
        "n_eval": 3000,
        "k": 10,
        "top_n": 3000,
    }

    _write_model_card(_toy_comparison(), "abc123def4567890", out, ctx)

    text = out.read_text(encoding="utf-8")
    assert "## Performance" in text
    assert "## Limitacoes" in text
    assert "## Possiveis Vieses" in text
    assert "coverage@10 = 0.05" in text  # promoted model's coverage
    assert "nao superou os baselines" in text  # ncf below baselines here
