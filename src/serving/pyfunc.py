"""MLflow pyfunc wrapper for the Production model (any of the 5 recommenders).

A single ``RecommenderPyfunc`` dispatches scoring to the model identified by
``model_type`` at registration time. This ensures the promoted model in the
MLflow Model Registry matches the actual best model from evaluation, not a
hardcoded one (edital §5: "Melhor modelo registrado").
"""

import json
import pickle
from typing import Any, Callable

import mlflow.pyfunc
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.evaluation.ranking import top_k_from_scores


def _artifacts_popularity(models_dir: Any, processed_dir: Any) -> dict[str, str]:
    """Assembles artifact paths for the popularity baseline."""
    return {"ranking": str(models_dir / "baseline_popularity" / "ranking.pkl")}


def _artifacts_item_cf(models_dir: Any, processed_dir: Any) -> dict[str, str]:
    """Assembles artifact paths for the item-based CF model."""
    return {
        "similarity": str(models_dir / "item_based_cf" / "item_similarity.npz"),
        "interactions": str(processed_dir / "interactions_prior.npz"),
    }


def _artifacts_user_cf(models_dir: Any, processed_dir: Any) -> dict[str, str]:
    """Assembles artifact paths for the user-based CF model."""
    return {
        "knn": str(models_dir / "user_based_cf" / "knn_model.pkl"),
        "interactions": str(processed_dir / "interactions_prior.npz"),
    }


def _artifacts_mf(models_dir: Any, processed_dir: Any) -> dict[str, str]:
    """Assembles artifact paths for the matrix factorization model."""
    return {
        "user_factors": str(models_dir / "matrix_factorization" / "user_factors.npy"),
        "item_factors": str(models_dir / "matrix_factorization" / "item_factors.npy"),
    }


def _artifacts_ncf(models_dir: Any, processed_dir: Any) -> dict[str, str]:
    """Assembles artifact paths for the NCF (neural network) model."""
    return {
        "model": str(models_dir / "neural_network" / "model.pt"),
        "metrics": str(models_dir / "neural_network" / "metrics.json"),
        "vocab": str(processed_dir / "vocabularies.pkl"),
    }


_ARTIFACT_BUILDERS: dict[str, Callable[..., dict[str, str]]] = {
    "popularity": _artifacts_popularity,
    "item_based_cf": _artifacts_item_cf,
    "user_based_cf": _artifacts_user_cf,
    "matrix_factorization": _artifacts_mf,
    "ncf": _artifacts_ncf,
}


def build_artifacts(
    model_type: str, models_dir: Any, processed_dir: Any
) -> dict[str, str]:
    """Assembles the artifact dict for the given model type.

    Args:
        model_type: One of the 5 registered model types.
        models_dir: Root models directory.
        processed_dir: The ``data/processed`` directory.

    Returns:
        Map artifact-name -> absolute path (str).
    """
    builder = _ARTIFACT_BUILDERS.get(model_type)
    if builder is None:
        raise ValueError(f"Unknown model_type '{model_type}'")
    return builder(models_dir, processed_dir)


def _load_pickle(path: str) -> Any:
    """Loads a pickle file and returns its content."""
    with open(path, "rb") as f:
        return pickle.load(f)


class _PopularityScorer:
    """Scorer returning the global popularity ranking as a score vector."""

    def __init__(self, artifacts: dict[str, str]) -> None:
        self._ranking = np.asarray(_load_pickle(artifacts["ranking"]))

    def __call__(self, user_idx: int) -> np.ndarray:
        scores = np.zeros(self._ranking.max() + 1, dtype=np.float32)
        scores[self._ranking] = np.arange(len(self._ranking), 0, -1, dtype=np.float32)
        return scores


class _ItemCFScorer:
    """Item-based CF scorer: user history dot item similarity."""

    def __init__(self, artifacts: dict[str, str]) -> None:
        self._similarity = sp.load_npz(artifacts["similarity"])
        self._interactions = sp.load_npz(artifacts["interactions"])

    def __call__(self, user_idx: int) -> np.ndarray:
        return self._interactions[user_idx].dot(self._similarity).toarray().flatten()


class _UserCFScorer:
    """User-based CF scorer: weighted sum of nearest neighbors' interactions."""

    def __init__(self, artifacts: dict[str, str]) -> None:
        bundle = _load_pickle(artifacts["knn"])
        self._knn = bundle["model"]
        self._pool_indices = bundle["pool_indices"]
        self._interactions = sp.load_npz(artifacts["interactions"])

    def __call__(self, user_idx: int) -> np.ndarray:
        pool = self._interactions[self._pool_indices]
        dist, neighbors = self._knn.kneighbors(self._interactions[user_idx])
        sims = (1.0 - dist).flatten()
        return (sims[:, None] * pool[neighbors.flatten()].toarray()).sum(axis=0)


class _MFScorer:
    """Matrix factorization scorer: user factor dot item factors."""

    def __init__(self, artifacts: dict[str, str]) -> None:
        self._user_factors = np.load(artifacts["user_factors"])
        self._item_factors = np.load(artifacts["item_factors"])

    def __call__(self, user_idx: int) -> np.ndarray:
        return self._user_factors[user_idx] @ self._item_factors.T


class _NCFScorer:
    """NCF scorer: neural network forward pass for all items (needs torch)."""

    def __init__(self, artifacts: dict[str, str]) -> None:
        import torch

        from src.models.ncf import NeuralCollaborativeFiltering

        with open(artifacts["metrics"], encoding="utf-8") as f:
            params = json.load(f)["params"]
        vocab = _load_pickle(artifacts["vocab"])
        self._n_items = len(vocab["idx_to_product_id"])
        model = NeuralCollaborativeFiltering(
            len(vocab["user_id_to_idx"]),
            self._n_items,
            params["embedding_dim"],
            tuple(params["hidden_dims"]),
        )
        model.load_state_dict(
            torch.load(artifacts["model"], map_location="cpu", weights_only=False)
        )
        model.eval()
        self._model = model

    def __call__(self, user_idx: int) -> np.ndarray:
        from src.models.ncf import score_all_items

        return score_all_items(self._model, [user_idx], self._n_items)[0]


_SCORER_CLASSES: dict[str, type] = {
    "popularity": _PopularityScorer,
    "item_based_cf": _ItemCFScorer,
    "user_based_cf": _UserCFScorer,
    "matrix_factorization": _MFScorer,
    "ncf": _NCFScorer,
}


def build_scorer(model_type: str, models_dir: Any, processed_dir: Any) -> Any:
    """Builds the scorer of a model type directly from on-disk artifacts.

    Lets the API serve whichever model the evaluation stage promoted, using the
    same scoring code that is packaged into the Registry — instead of a
    hardcoded recommender.

    Args:
        model_type: One of the 5 registered model types.
        models_dir: Root models directory.
        processed_dir: The ``data/processed`` directory.

    Returns:
        Callable scorer exposing ``__call__(user_idx) -> np.ndarray``.

    Raises:
        ValueError: If model_type is not one of the 5 known types.
    """
    cls = _SCORER_CLASSES.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model_type '{model_type}'")
    return cls(build_artifacts(model_type, models_dir, processed_dir))


class RecommenderPyfunc(mlflow.pyfunc.PythonModel):
    """Serves any of the 5 recommenders from packaged MLflow artifacts.

    The ``model_type`` is set at construction time (before ``log_model``) and
    determines which scoring path is used at inference time.

    Args:
        model_type: One of the 5 registered model types.
    """

    def __init__(self, model_type: str = "item_based_cf") -> None:
        self._model_type = model_type
        self._scorer: Callable[[int], np.ndarray] | None = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Builds the scorer from the artifacts for the configured model type."""
        cls = _SCORER_CLASSES.get(self._model_type)
        if cls is None:
            raise ValueError(f"Unknown model_type '{self._model_type}'")
        self._scorer = cls(context.artifacts)

    def score_user(self, user_idx: int) -> np.ndarray:
        """Computes the dense per-item score vector for a known user.

        Args:
            user_idx: Internal user index.

        Returns:
            Numpy score vector, shape (n_items,).
        """
        return self._scorer(user_idx)

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
