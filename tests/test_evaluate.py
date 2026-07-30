"""Tests for the shared evaluation-population sampling (comparison table)."""

from src.pipeline.evaluate import _sample_users


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
