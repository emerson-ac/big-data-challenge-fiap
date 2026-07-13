"""Preprocessing for recommendation interactions.

Strategy pattern (Aula 03): each strategy implements the same interface so the
preprocessing pipeline can swap or chain them at runtime without changing the
caller. See ``docs/design-pattern.md``.
"""

from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.strategies import (
    InteractionFilterStrategy,
    PreprocessingStrategy,
    UserItemEncoderStrategy,
)

__all__ = [
    "InteractionFilterStrategy",
    "PreprocessingStrategy",
    "Preprocessor",
    "UserItemEncoderStrategy",
]
