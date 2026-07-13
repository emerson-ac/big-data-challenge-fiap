"""MLflow pyfunc wrapper for the Production model (item-based CF)."""

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.evaluation.ranking import top_k_from_scores


class ItemBasedCFPyfunc(mlflow.pyfunc.PythonModel):
    """Serves item-based CF recommendations from packaged artifacts.

    Receives a DataFrame with ``user_idx`` and ``k`` columns and returns, per
    row, the list of recommended item_idx values.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Loads similarity and interactions from the model artifacts."""
        self._similarity = sp.load_npz(context.artifacts["similarity"])
        self._interactions = sp.load_npz(context.artifacts["interactions"])

    def score_user(self, user_idx: int) -> np.ndarray:
        """Computes the dense per-item score vector for a known user.

        Args:
            user_idx: Internal user index (row of the interaction matrix).

        Returns:
            Numpy score vector, shape (n_items,).
        """
        scores = self._interactions[user_idx].dot(self._similarity)
        return scores.toarray().flatten()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> list[list[int]]:
        """Generates the top-k of item_idx for each user_idx in the input.

        Args:
            context: pyfunc context (unused after load).
            model_input: DataFrame with user_idx and k columns.
            params: Inference params (unused).

        Returns:
            List of lists of recommended item_idx.
        """
        return [
            top_k_from_scores(self.score_user(int(row["user_idx"])), int(row["k"]))
            for _, row in model_input.iterrows()
        ]


def build_artifacts(models_dir: Any, processed_dir: Any) -> dict[str, str]:
    """Assembles the artifact dict required by the pyfunc.

    Args:
        models_dir: Root models directory.
        processed_dir: The ``data/processed`` directory.

    Returns:
        Map artifact-name -> absolute path (str).
    """
    return {
        "similarity": str(models_dir / "item_based_cf" / "item_similarity.npz"),
        "interactions": str(processed_dir / "interactions_prior.npz"),
    }
