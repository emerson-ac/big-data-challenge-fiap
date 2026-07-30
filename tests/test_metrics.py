"""Tests for the ranking metrics that decide which model is promoted.

Expected values are computed by hand in each docstring, so a regression in the
formulas fails here instead of silently changing the promoted model.
"""

import math

import pandas as pd

from src.evaluation.metrics import (
    average_precision_at_k,
    coverage_at_k,
    evaluate_recommendations,
    hit_rate_at_k,
    ndcg_at_k,
    pairs_to_ground_truth,
    precision_at_k,
    recall_at_k,
)


def test_precision_counts_only_hits_within_k() -> None:
    """2 hits in a top-4 list gives 2/4."""
    assert precision_at_k([1, 2, 3, 4], {2, 4}, 4) == 0.5


def test_precision_ignores_hits_beyond_k() -> None:
    """The hit at position 3 is outside the top-2, so precision is 0."""
    assert precision_at_k([1, 2, 3], {3}, 2) == 0.0


def test_precision_is_zero_for_k_zero() -> None:
    """k=0 short-circuits instead of dividing by zero."""
    assert precision_at_k([1, 2], {1}, 0) == 0.0


def test_recall_divides_by_relevant_set_size() -> None:
    """2 of the 3 relevant items retrieved gives 2/3."""
    assert recall_at_k([1, 2, 3], {2, 3, 5}, 3) == 2 / 3


def test_recall_is_zero_without_ground_truth() -> None:
    """An empty relevant set yields 0 rather than a division error."""
    assert recall_at_k([1, 2], set(), 2) == 0.0


def test_ndcg_discounts_by_position() -> None:
    """Single hit at position 2: dcg=1/log2(3), idcg=1/log2(2)=1."""
    assert ndcg_at_k([1, 2, 3], {2}, 3) == 1.0 / math.log2(3)


def test_ndcg_is_one_for_perfect_ranking() -> None:
    """Every relevant item on top gives dcg == idcg."""
    assert ndcg_at_k([1, 2], {1, 2}, 2) == 1.0


def test_ndcg_penalizes_worse_position() -> None:
    """The same hit ranked lower must score strictly less."""
    assert ndcg_at_k([9, 1], {1}, 2) < ndcg_at_k([1, 9], {1}, 2)


def test_average_precision_averages_precision_at_each_hit() -> None:
    """Hits at positions 2 and 4: (1/2 + 2/4) / min(2, 4) = 0.5."""
    assert average_precision_at_k([1, 2, 3, 4], {2, 4}, 4) == 0.5


def test_average_precision_rewards_earlier_hits() -> None:
    """Same number of hits, earlier positions must score higher."""
    early = average_precision_at_k([1, 2, 9, 9], {1, 2}, 4)
    late = average_precision_at_k([9, 9, 1, 2], {1, 2}, 4)

    assert early == 1.0
    assert early > late


def test_hit_rate_counts_users_with_at_least_one_hit() -> None:
    """User 0 hits, user 1 does not: 1 of 2 users."""
    recs = {0: [1, 2], 1: [7, 8]}
    ground_truth = {0: {2}, 1: {3}}

    assert hit_rate_at_k(recs, ground_truth, 2) == 0.5


def test_hit_rate_is_zero_without_ground_truth() -> None:
    """No users to evaluate yields 0 instead of a division error."""
    assert hit_rate_at_k({0: [1]}, {}, 1) == 0.0


def test_coverage_counts_distinct_items_over_catalog() -> None:
    """Items {1, 2, 3} recommended out of a 10-item catalog gives 0.3."""
    recs = {0: [1, 2], 1: [2, 3]}

    assert coverage_at_k(recs, 10, 2) == 0.3


def test_coverage_ignores_items_beyond_k() -> None:
    """Only the top-1 of each user counts, so items {1, 2} of 10."""
    recs = {0: [1, 9], 1: [2, 8]}

    assert coverage_at_k(recs, 10, 1) == 0.2


def test_coverage_is_zero_for_empty_catalog() -> None:
    """n_items=0 short-circuits instead of dividing by zero."""
    assert coverage_at_k({0: [1]}, 0, 1) == 0.0


def test_pairs_to_ground_truth_groups_items_per_user() -> None:
    """Positive pairs become one set of relevant items per user."""
    pairs = pd.DataFrame(
        {"user_idx": [0, 0, 1], "item_idx": [10, 11, 12]},
    )

    assert pairs_to_ground_truth(pairs) == {0: {10, 11}, 1: {12}}


def test_evaluate_recommendations_returns_the_four_official_metrics() -> None:
    """The aggregate exposes exactly the metrics the comparison table uses."""
    recs = {0: [1, 2]}
    ground_truth = {0: {1}}

    metrics = evaluate_recommendations(recs, ground_truth, 2)

    assert set(metrics) == {
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "map_at_k",
    }


def test_evaluate_recommendations_averages_over_users() -> None:
    """User 0 has perfect recall and user 1 none, averaging to 0.5."""
    recs = {0: [1], 1: [9]}
    ground_truth = {0: {1}, 1: {2}}

    metrics = evaluate_recommendations(recs, ground_truth, 1)

    assert metrics["recall_at_k"] == 0.5


def test_evaluate_recommendations_penalizes_missing_users() -> None:
    """A user absent from the recommendations scores 0, it is not skipped."""
    ground_truth = {0: {1}, 1: {2}}

    metrics = evaluate_recommendations({0: [1]}, ground_truth, 1)

    assert metrics["recall_at_k"] == 0.5
