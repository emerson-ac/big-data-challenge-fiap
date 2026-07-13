# Design Pattern

## Resumo

Documentação dos padrões de projeto implementados no projeto.

- **Arquitetura:** Monolítica
- **Paradigma:** OOP (Object Oriented Programming)
- **Padrões:**
  - Factory: criação/carregamento de modelos
  - Strategy: pré-processadores
- **Lint:** ruff
- **Commits:** pre-commit hooks (`uv run pre-commit run --all-files`)

---

## Criação de modelos (Factory Pattern)

**Problema:** o pipeline precisa instanciar/carregar diferentes recomendadores
(item-based CF, popularidade) sem que o código cliente conheça a classe concreta
de cada um.

**Solução:** `ModelFactory` centraliza o registro e a criação. Novos modelos são
adicionados via `ModelFactory.register(...)`, sem alterar o código que consome a
factory (princípio Open/Closed).

```python
class ModelFactory:
    """Factory para criar/carregar modelos de recomendação."""

    _builders: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str, builder: Callable[..., Any]) -> None:
        """Registra um builder (classe ou callable) sob um nome."""
        cls._builders[name] = builder

    @classmethod
    def create(cls, model_type: str, **kwargs: Any) -> Any:
        """Cria/carrega um modelo registrado pelo nome."""
        builder = cls._builders.get(model_type)
        if builder is None:
            raise ValueError(f"Modelo '{model_type}' não registrado")
        return builder(**kwargs)


ModelFactory.register("item_based_cf", ItemBasedCFRecommender.load)
ModelFactory.register("popularity", PopularityRecommender.load)
```

**Onde é usado:** `src/models/model_loader.py`, consumido por
`src/models/inference.py` (`RecommendationEngine`) e pelos testes em
`tests/test_model_loader.py`.

---

## Pré-processadores (Strategy Pattern)

**Problema:** diferentes etapas exigem transformações distintas sobre o frame de
interações (filtrar catálogo popular, codificar ids em índices). Lógica
condicional (`if etapa == ...`) cresceria a cada nova transformação, violando
Open/Closed.

**Solução:** cada estratégia implementa a interface comum `PreprocessingStrategy`
(Protocol) e o contexto `Preprocessor` as encadeia em ordem. Trocar ou reordenar
estratégias não altera o chamador.

```python
class PreprocessingStrategy(Protocol):
    """Interface comum para estratégias de pré-processamento."""

    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Aplica a transformação e retorna o frame processado."""
        ...


class InteractionFilterStrategy:
    """Mantém os produtos mais populares e os usuários mais ativos."""
    ...


class UserItemEncoderStrategy:
    """Codifica ids brutos em índices contíguos e constrói o vocabulário."""
    ...


class Preprocessor:
    """Contexto que aplica a sequência de estratégias configurada."""

    def __init__(self, strategies: Sequence[PreprocessingStrategy]) -> None:
        self._strategies = list(strategies)

    def run(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Aplica cada estratégia em sequência e retorna o frame final."""
        ...
```

O vocabulário construído por `UserItemEncoderStrategy`
(`user_id_to_idx`, `idx_to_product_id`) é compatível com os artefatos consumidos
por `src/models/inference.py`.

**Onde é usado:** `src/preprocessing/` (`strategies.py`, `preprocessor.py`),
consumido pelos notebooks de pré-processamento e testado em
`tests/test_preprocessing.py`.

---

## Referências

- [naming-conventions.md](naming-conventions.md) — Convenções de nomenclatura e SOLID
- [REQUIREMENTS.md](REQUIREMENTS.md) — Checklist de requisitos obrigatórios
- Aula 03 — Padrões de Projeto em ML (Clean Code, Fase 2)
