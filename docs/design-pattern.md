# Design Pattern

## Summary
Documentation about the design pattern definitions used in this project.

- **Architecture:** Monolítica
- **Programming paradigm:** OOP (Object Oriented Programming)
- **Design Patterns:**
  - Factory: criação de modelos
  - Strategy: pré-processadores
- **Lint:** ruff
- **Commits:** Pre-commit hooks configurados por uvx (`uvx pre-commit run --all-files`)

---

## Models creation (Factory Pattern)

**Problema:** o pipeline precisa instanciar diferentes tipos de modelo (baselines Scikit-Learn — popularidade, KNN, co-ocorrência — e o modelo neural PyTorch) sem que o código cliente (`recommendation_service.py`, notebooks de treino/avaliação) conheça a classe concreta de cada um.

**Solução:** `ModelFactory` centraliza o registro e a criação de modelos. Novos modelos são adicionados via `ModelFactory.register(...)`, sem alterar o código que já consome a factory (princípio Open/Closed).

```python
class ModelFactory:
    """Factory para criação de modelos de recomendação."""

    _models: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """Registra um novo tipo de modelo na factory."""
        cls._models[name] = model_class

    @classmethod
    def create(cls, model_type: str, **kwargs) -> object:
        """Instancia um modelo registrado pelo nome."""
        model_class = cls._models.get(model_type)
        if model_class is None:
            raise ValueError(f"Modelo '{model_type}' não registrado")
        return model_class(**kwargs)


ModelFactory.register("popularity", PopularityRecommender)
ModelFactory.register("knn", KNNRecommender)
ModelFactory.register("neural_network", NeuralNetworkModel)
```

**Onde é usado:** `src/models/model_loader.py`, consumido por `RecommendationService` e pelos notebooks `02_baseline.ipynb` e `04_model_training.ipynb`.

---

## Pre-processors (Strategy Pattern)

**Problema:** diferentes etapas/modelos exigem pré-processamentos distintos (normalização para a rede neural, encoding de usuário/item para embeddings, agregações RFM para baselines). Lógica condicional (`if model_type == ...`) cresceria a cada novo modelo, violando Open/Closed.

**Solução:** cada estratégia de pré-processamento implementa uma interface comum (`PreprocessingStrategy`), e o componente que a usa recebe a estratégia por injeção, podendo trocá-la em tempo de execução.

```python
from typing import Protocol
import pandas as pd


class PreprocessingStrategy(Protocol):
    """Interface comum para estratégias de pré-processamento."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aplica a transformação e retorna o DataFrame processado."""
        ...


class RFMFeatureStrategy:
    """Gera features de recência, frequência e valor monetário."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data  # implementação específica


class EmbeddingEncodingStrategy:
    """Codifica usuários e itens para índices de embedding."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data  # implementação específica


class Preprocessor:
    """Contexto que aplica a estratégia configurada."""

    def __init__(self, strategy: PreprocessingStrategy) -> None:
        self._strategy = strategy

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._strategy.transform(data)
```

**Onde é usado:** `src/api/services/preprocessing_service.py`, consumido pelos notebooks `02_baseline.ipynb` (RFM) e `03_preprocessing.ipynb`/`04_model_training.ipynb` (encoding de embeddings).

---

## References
- [naming-conventions.md](naming-conventions.md) - Convenções de nomenclatura e SOLID
- [REQUIREMENTS.md](REQUIREMENTS.md) - Checklist de requisitos obrigatórios
