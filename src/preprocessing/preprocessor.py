"""Preprocessor context that chains preprocessing strategies."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.preprocessing.strategies import PreprocessingStrategy


class Preprocessor:
    """Applies an ordered sequence of ``PreprocessingStrategy`` instances.

    Args:
        strategies: Strategies applied in order over the interaction frame.
    """

    def __init__(self, strategies: Sequence[PreprocessingStrategy]) -> None:
        self._strategies = list(strategies)

    def run(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Runs every strategy in sequence and returns the final frame.

        Args:
            interactions: Raw interaction frame.

        Returns:
            Fully preprocessed interaction frame.
        """
        processed = interactions
        for strategy in self._strategies:
            processed = strategy.transform(processed)
        return processed
