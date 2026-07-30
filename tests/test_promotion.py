"""Tests for the model promotion rule (single source of truth)."""

from pathlib import Path

import pandas as pd

from src.evaluation.promotion import promoted_model_from_csv, select_promoted_model


def _comparison(recalls: dict[str, float]) -> pd.DataFrame:
    """Builds a comparison table indexed by model name."""
    return pd.DataFrame({"recall_at_k": recalls}).rename_axis("model")


def test_select_promoted_model_picks_highest_recall() -> None:
    """The promoted model is the one with the highest recall@k."""
    df = _comparison({"popularity": 0.30, "item_based_cf": 0.42, "ncf": 0.41})

    assert select_promoted_model(df) == "item_based_cf"


def test_select_promoted_model_breaks_ties_by_first_occurrence() -> None:
    """On a recall tie, idxmax keeps the first model (deterministic)."""
    df = _comparison({"popularity": 0.50, "item_based_cf": 0.50})

    assert select_promoted_model(df) == "popularity"


def test_promoted_model_from_csv_round_trips(tmp_path: Path) -> None:
    """Reading the comparison CSV yields the same promoted model."""
    csv_path = tmp_path / "metrics_comparison.csv"
    _comparison({"popularity": 0.30, "item_based_cf": 0.55}).to_csv(csv_path)

    assert promoted_model_from_csv(csv_path) == "item_based_cf"


def test_promoted_model_from_csv_missing_file_returns_none(tmp_path: Path) -> None:
    """A missing comparison CSV returns None so callers can fall back."""
    assert promoted_model_from_csv(tmp_path / "absent.csv") is None
