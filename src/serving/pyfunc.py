"""Wrapper MLflow pyfunc do modelo Production (item-based CF) para o Registry."""

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.evaluation.ranking import top_k_from_scores


class ItemBasedCFPyfunc(mlflow.pyfunc.PythonModel):
    """Serve recomendações item-based CF a partir de artefatos empacotados.

    Recebe um DataFrame com as colunas ``user_idx`` e ``k`` e devolve, por
    linha, a lista de item_idx recomendados.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Carrega similaridade e interações dos artefatos do modelo."""
        self._similarity = sp.load_npz(context.artifacts["similarity"])
        self._interactions = sp.load_npz(context.artifacts["interactions"])

    def score_user(self, user_idx: int) -> np.ndarray:
        """Calcula o vetor denso de scores por item para um usuário.

        Args:
            user_idx: Índice interno do usuário (linha das interações).

        Returns:
            Vetor numpy de scores, shape (n_items,).
        """
        scores = self._interactions[user_idx].dot(self._similarity)
        return scores.toarray().flatten()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> list[list[int]]:
        """Gera o top-k de item_idx para cada user_idx do input.

        Args:
            context: Contexto pyfunc (não utilizado após load).
            model_input: DataFrame com colunas user_idx e k.
            params: Parâmetros de inferência (não utilizados).

        Returns:
            Lista de listas de item_idx recomendados.
        """
        return [
            top_k_from_scores(self.score_user(int(row["user_idx"])), int(row["k"]))
            for _, row in model_input.iterrows()
        ]


def build_artifacts(models_dir: Any, processed_dir: Any) -> dict[str, str]:
    """Monta o dicionário de artefatos exigido pelo pyfunc.

    Args:
        models_dir: Diretório raiz dos modelos.
        processed_dir: Diretório ``data/processed``.

    Returns:
        Mapa nome-do-artefato -> caminho absoluto (str).
    """
    return {
        "similarity": str(models_dir / "item_based_cf" / "item_similarity.npz"),
        "interactions": str(processed_dir / "interactions_prior.npz"),
    }
