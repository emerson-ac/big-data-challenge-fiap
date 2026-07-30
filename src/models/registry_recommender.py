"""Recommender that loads the Production model from the MLflow Model Registry.

Alternative to ``ItemBasedCFRecommender`` (which reads artifacts from disk):
resolves the model by alias ``@production`` in the MLflow server and exposes
the same ``score_user`` interface — the inference engine treats both equally.
"""

import mlflow
import numpy as np
import structlog

from src.config import get_settings

logger = structlog.get_logger()


class RegistryRecommender:
    """Adapts the Registry pyfunc model to the engine's scoring interface.

    Args:
        pyfunc_impl: Loaded ``ItemBasedCFPyfunc`` instance (with
            similarity/interactions), obtained via ``unwrap_python_model``.
    """

    def __init__(self, pyfunc_impl: object) -> None:
        self._impl = pyfunc_impl

    @classmethod
    def load(cls, name: str, alias: str) -> "RegistryRecommender":
        """Loads ``models:/<name>@<alias>`` from MLflow and unwraps the pyfunc.

        Args:
            name: Registered model name.
            alias: Promoted alias (e.g. ``production``).

        Returns:
            Instance ready to produce scores.
        """
        mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
        uri = f"models:/{name}@{alias}"
        model = mlflow.pyfunc.load_model(uri)
        logger.info("registry_model_loaded", uri=uri)
        return cls(model.unwrap_python_model())

    @property
    def model_type(self) -> str | None:
        """Base model type actually served by the promoted pyfunc.

        Read from the unwrapped ``RecommenderPyfunc`` (e.g. ``popularity``),
        so the API can report the real model instead of a hardcoded name.
        """
        return getattr(self._impl, "_model_type", None)

    def score_user(self, user_idx: int) -> np.ndarray:
        """Computes the dense per-item score vector for a known user.

        Args:
            user_idx: Internal user index.

        Returns:
            Numpy score vector, shape (n_items,).
        """
        return self._impl.score_user(user_idx)
