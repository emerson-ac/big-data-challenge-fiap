"""Predictor KServe customizado que serve o RecommendationEngine (item-based CF).

Diferente do modelo Iris (formato sklearn nativo), o recomendador não é um
``modelFormat`` padrão do KServe — por isso sobe como *custom container*.
Expõe o protocolo V1 do KServe em ``/v1/models/recsys:predict``.

Payload esperado:
    {"instances": [{"user_id": 123, "k": 10}, {"user_id": 456}]}
"""

import argparse

from kserve import Model, ModelServer

from src.config import get_settings
from src.models.inference import RecommendationEngine


class RecsysModel(Model):
    """Modelo KServe que delega ao RecommendationEngine do modelo Production."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
        self._engine: RecommendationEngine | None = None
        self.load()

    def load(self) -> None:
        """Carrega o engine (modelo Production) e marca o serviço como pronto."""
        self._engine = RecommendationEngine(model_type=get_settings().model_type)
        self.ready = True

    def predict(self, payload: dict, headers: dict | None = None) -> dict:
        """Gera o top-k de recomendações para cada instância do payload.

        Args:
            payload: Dicionário com a chave ``instances``.
            headers: Cabeçalhos HTTP (não utilizados).

        Returns:
            Dicionário no formato ``{"predictions": [...]}`` do protocolo V1.
        """
        predictions = [self._predict_one(inst) for inst in payload["instances"]]
        return {"predictions": predictions}

    def _predict_one(self, instance: dict) -> list[dict]:
        """Recomenda para uma única instância (user_id + k opcional)."""
        recs = self._engine.recommend(int(instance["user_id"]), int(instance.get("k", 10)))
        return [
            {"product_id": r.product_id, "score": r.score, "rank": r.rank} for r in recs
        ]


def main() -> None:
    """Sobe o servidor KServe com o modelo de recomendação."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="recsys")
    args, _ = parser.parse_known_args()
    ModelServer().start([RecsysModel(args.model_name)])


if __name__ == "__main__":
    main()
