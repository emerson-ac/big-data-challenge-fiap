"""Model promotion rule: the single source of truth for choosing "best".

Both the evaluation pipeline (which promotes to the MLflow Registry) and the
API (which selects the model to serve in local mode) resolve the promoted
model through these functions, so the two can never drift apart.
"""

from pathlib import Path

import pandas as pd

_PROMOTION_METRIC = "recall_at_k"


def select_promoted_model(comparison: pd.DataFrame) -> str:
    """Returns the model promoted to Production (highest recall@k).

    Args:
        comparison: Comparison table indexed by model name, with a
            ``recall_at_k`` column.

    Returns:
        Name of the promoted model.
    """
    return str(comparison[_PROMOTION_METRIC].idxmax())


def promoted_model_from_csv(path: Path) -> str | None:
    """Reads the comparison CSV and returns the promoted model name.

    Args:
        path: Path to ``metrics_comparison.csv``.

    Returns:
        The promoted model name, or ``None`` if the file does not exist.
    """
    if not path.exists():
        return None
    comparison = pd.read_csv(path, index_col="model")
    return select_promoted_model(comparison)
