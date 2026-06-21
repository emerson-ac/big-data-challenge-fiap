# Name Convention and SOLID

## Summary
This is ML project designed upon Clean Code and SOLID principles.
Read this documentation if you want to know about the naming conventions used in this project.

## Conventions

### Files and Modules

- Snake_case para nomes de arquivos e módulos: `data_loader.py`, `model_factory.py`.
- Um módulo por responsabilidade (ex.: `preprocessing.py` não deve conter lógica de treinamento).
- Notebooks numerados por ordem de execução: `01_eda.ipynb`, `02_baseline.ipynb` (ver [NOTEBOOKS.md](NOTEBOOKS.md)).

### Classes

- PascalCase: `RecommendationService`, `ModelFactory`, `PreprocessingStrategy`.
- Sufixo indica o papel: `*Service` (lógica de negócio), `*Factory` (criação de objetos), `*Strategy` (algoritmo intercambiável), `*Repository` (acesso a dados).

### Funções e Variáveis

- snake_case, verbos no infinitivo para funções: `load_dataset()`, `compute_metrics()`, `train_model()`.
- Booleanos com prefixo `is_`/`has_`/`should_`: `is_valid_user`, `has_missing_values`.
- Evitar abreviações ambíguas (`df` é aceitável para DataFrame; `usr`, `itm` não são).

### Constantes

- UPPER_SNAKE_CASE, definidas no topo do módulo ou em `configs/`.
- `RANDOM_SEED = 42` deve ser a única fonte de verdade para seed em todo o projeto — nunca hardcode `42` espalhado pelo código; importar a constante.

### Type Hints e Docstrings

- Type hints obrigatórios em toda função pública (parâmetros e retorno).
- Docstrings no padrão Google Style, descrevendo `Args`, `Returns` e `Raises` quando aplicável.
- Funções com no máximo 20 linhas; extrair sub-funções privadas (prefixo `_`) quando exceder.

```python
def compute_precision_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Calcula precision@k para uma lista de recomendações.

    Args:
        recommended: Itens recomendados, ordenados por score.
        relevant: Conjunto de itens relevantes (ground truth).
        k: Número de itens considerados no top-k.

    Returns:
        Valor de precision@k entre 0 e 1.
    """
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k
```

## Seeds

- `RANDOM_SEED = 42` fixado em **todo** código que usa aleatoriedade: `numpy`, `random`, `torch`, `sklearn` (`train_test_split`, `StratifiedKFold`, etc.).
- Seed deve ser fixada uma única vez por execução (script/notebook), em uma função de setup centralizada (ex.: `set_seed(seed: int) -> None`), nunca de forma implícita ou duplicada.

## Logging

- Proibido `print()` em qualquer módulo de `src/` — usar `structlog`. Em notebooks, `print()` é permitido (uso exploratório).
- Eventos de log nomeados em snake_case descrevendo o que ocorreu, não como: `model_training_started`, `recommendation_generated`, não `"Treinando modelo..."`.
- Contexto relevante (ids, métricas, parâmetros) deve ser passado como kwargs estruturados, não interpolado na mensagem.

```python
logger.info("model_training_started", model_type="mlp", epochs=epochs, lr=learning_rate)
```

## Commits

- Seguir Conventional Commits — ver [COMMIT-CONVENTIONS.md](COMMIT-CONVENTIONS.md) para a especificação completa.

## SOLID neste Projeto

- **S (Single Responsibility):** cada classe/módulo tem um único motivo para mudar (ex.: `RecommendationService` não lida com parsing de request).
- **O (Open/Closed):** novos modelos/preprocessadores são adicionados via `ModelFactory`/`Strategy`, sem alterar código existente.
- **L (Liskov Substitution):** qualquer implementação de `PreprocessingStrategy` deve ser intercambiável sem quebrar o chamador.
- **I (Interface Segregation):** interfaces pequenas e específicas (ex.: `Predictable`, `Trainable`) em vez de uma interface genérica de "modelo".
- **D (Dependency Inversion):** serviços dependem de abstrações (`ModelFactory.create(...)`), não de classes concretas.

## References
- Clean Code: Aula 1 - Boas Práticas de Código Limpo em ML
- Clean Code: Aula 2 - Refatoração de Código e Qualidade em ML
- [design-pattern.md](design-pattern.md) - Padrões de projeto implementados
- [REQUIREMENTS.md](REQUIREMENTS.md) - Checklist de requisitos obrigatórios
