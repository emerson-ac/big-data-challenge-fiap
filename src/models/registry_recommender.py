"""Recomendador que carrega o modelo Production do MLflow Model Registry.

Alternativa ao ``ItemBasedCFRecommender`` (que lê artefatos do disco): resolve o
modelo pelo alias ``@production`` no servidor MLflow e expõe a mesma interface
``score_user`` — o engine de inferência trata ambos de forma idêntica.
"""

import mlflow
import numpy as np
import structlog

from src.config import get_settings

logger = structlog.get_logger()


class RegistryRecommender:
    """Adapta o modelo pyfunc do Registry à interface de scoring do engine.

    Args:
        pyfunc_impl: Instância do ``ItemBasedCFPyfunc`` já carregada (com
            similaridade/interações), obtida via ``unwrap_python_model``.
    """

    def __init__(self, pyfunc_impl: object) -> None:
        self._impl = pyfunc_impl

    @classmethod
    def load(cls, name: str, alias: str) -> "RegistryRecommender":
        """Carrega ``models:/<name>@<alias>`` do MLflow e desembrulha o pyfunc.

        Args:
            name: Nome do modelo registrado.
            alias: Alias promovido (ex.: ``production``).

        Returns:
            Instância pronta para gerar scores.
        """
        mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
        uri = f"models:/{name}@{alias}"
        model = mlflow.pyfunc.load_model(uri)
        logger.info("registry_model_loaded", uri=uri)
        return cls(model.unwrap_python_model())

    def score_user(self, user_idx: int) -> np.ndarray:
        """Calcula o vetor denso de scores por item para um usuário conhecido.

        Args:
            user_idx: Índice interno do usuário.

        Returns:
            Vetor numpy de scores, shape (n_items,).
        """
        return self._impl.score_user(user_idx)
