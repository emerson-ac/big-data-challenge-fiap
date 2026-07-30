# Sistema de Recomendação de Produtos - Tech Challenge 2

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138+-green)
![DVC](https://img.shields.io/badge/DVC-3.67+-purple)
![MLflow](https://img.shields.io/badge/MLflow-2.22+-orange)
![Docker](https://img.shields.io/badge/Docker-multi--stage-blue)

## Visão Geral

Sistema de recomendação de produtos para e-commerce baseado no comportamento de compra de usuários. Compara 5 modelos (popularidade, item-based CF, user-based CF, matrix factorization e rede neural NCF em PyTorch), com pipeline reproduzível via DVC, experimentos rastreados no MLflow, modelo servido via Model Registry e API REST (FastAPI) containerizada com Docker.

**Dataset:** [Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)

---

## Status do Projeto

- [x] **Etapa 1** — Clean Code e Estrutura (SOLID, Factory + Strategy, ≤20 linhas/fn)
- [x] **Etapa 2** — Ambiente e Dependências (uv, Pydantic Settings, validate_env)
- [x] **Etapa 3** — DVC + MLflow + Docker (pipeline reproduzível, tracking, container)
- [x] **Etapa 4** — Model Registry + Serving + CI (alias @production, FastAPI, GitHub Actions)

Checklist completo em [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md).

---

## Quick Start

```bash
git clone https://github.com/emerson-ac/big-data-challenge-fiap.git
cd big-data-challenge-fiap

uv sync                                     # instala ambiente reproduzível (lock)
cp .env.example .env                        # configura variáveis de ambiente
uv run python scripts/validate_env.py        # valida Python, deps, seed e .env
uv run pre-commit install                   # git hooks (ruff + uv-lock)
```

---

## Pipeline DVC (reproduzível)

O pipeline tem 3 estágios: `preprocess -> train -> evaluate`, declarados em
[`dvc.yaml`](dvc.yaml) e travados em [`dvc.lock`](dvc.lock).

### Dados

O dataset real do Instacart (680MB) é versionado pelo DVC (`data/raw.dvc`) e
mora no remote S3, não no git. Numa nova clonagem:

```bash
uv run dvc pull                              # baixa data/raw/ do S3 (~680MB)
```

Sem acesso ao bucket, baixe do
[Kaggle](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)
ou gere o dataset sintético:

```bash
uv run python scripts/gen_synthetic_data.py  # dataset sintético em data/raw/ (segundos)
```

O gerador remove o `data/raw.dvc` ao sobrescrever `data/raw/`, para que um
`dvc repro` não registre os dados sintéticos como o dataset do projeto. Ele
imprime como reverter.

> **Atenção:** os artefatos do pipeline (`data/processed/`, `models/`) também são
> gerenciados pelo DVC (cache padrão) e ignorados pelo git — use `dvc pull` para
> obtê-los e `dvc push` após um `dvc repro`.

### Executar

```bash
uv run dvc repro                              # preprocess -> train -> evaluate
uv run dvc status                             # deve estar "up to date"
```

### Remote DVC (versionamento de dados)

O remote default é o S3 (Aula 3 — Armazenamento Remoto), declarado em
[`.dvc/config`](.dvc/config):

```
s3://<seu-bucket>/dvc   (us-east-1)
```

```bash
uv run dvc push                               # envia dados/artefatos ao S3
uv run dvc pull                               # baixa dados/artefatos do S3
uv run dvc status -c                          # compara workspace x remote
```

Requer credenciais AWS com acesso ao bucket (`aws configure` ou SSO). Para
recriar a configuração do zero: `bash scripts/setup_dvc_s3.sh`.

---

## MLflow (tracking e Model Registry)

Cada estágio do pipeline rastreia parâmetros, métricas e artefatos no MLflow.
O experimento é namespaced como `recsys-instacart/*` (≥7 runs).

```bash
uv run mlflow ui --backend-store-uri mlruns   # UI em http://localhost:5000
```

O estágio `evaluate` registra o melhor modelo (vencedor por `Recall@10`) sob o
nome model-agnostic `recsys_recommender` no Model Registry, promovido via stages
`Staging -> Production` e alias `@production`. O pyfunc pode servir qualquer um
dos 5 modelos — o nome do registro não presume o algoritmo vencedor.

---

## API REST (FastAPI)

### Local

```bash
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
# Swagger: http://localhost:8000/docs
# Health:  http://localhost:8000/health/status
```

Exemplo de requisição:

```bash
curl -X POST http://localhost:8000/recommendations/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 5}'
```

A API carrega o modelo do disco (`MODEL_SOURCE=local`, default) ou do MLflow
Model Registry (`MODEL_SOURCE=registry`), com fallback de popularidade para
cold-start.

### Docker

> **Pré-requisito:** inicie o Docker (ou Colima no macOS: `colima start`)
> antes de construir a imagem.

```bash
docker build -t recsys-api .
docker run -p 8000:8000 recsys-api
# ou orquestração completa (MLflow + treino + API):
docker compose up --build
```

O `docker-compose.yml` sobe três serviços encadeados: **`mlflow`** (tracking
server), **`train`** (serviço de treino — gera dados sintéticos e roda o
pipeline `preprocess → train → evaluate`, registrando e promovendo o melhor
modelo no Registry) e **`api`** (serve o modelo `@production` do Registry,
lendo vocabulário e fallback dos volumes compartilhados). O `train` e a `api`
compartilham os volumes `models_data`/`processed_data`.

O `Dockerfile` é multi-stage (builder + runtime), `python:3.12-slim`, usuário
não-root, com healthcheck.

---

## Desenvolvimento

### Linting, Formatação e Testes

```bash
uv run ruff check .                # lint
uv run ruff format .               # formatação
uv run pytest -q                   # testes (45 testes)
uv run pre-commit run --all-files  # todos os hooks
```

### Commits

Padrão Conventional Commits (ver [`docs/COMMIT-CONVENTIONS.md`](docs/COMMIT-CONVENTIONS.md)):

```bash
git commit -m "feat: implementar modelo neural"
git commit -m "fix: corrigir normalizacao de features"
```

---

## Estrutura do Projeto

```
src/
├── api/                # API REST (FastAPI)
│   ├── main.py          # Aplicação (lifespan, exception handlers, rotas)
│   ├── config.py        # Settings da API (Pydantic)
│   ├── dependencies.py  # Injeção de dependências
│   ├── routes/          # health, recommendations
│   ├── schemas/         # request, response, errors
│   ├── services/        # recommendation_service
│   ├── middleware/      # error_handler
│   └── utils/           # logger (structlog)
├── config.py            # Settings globais (Pydantic + validação de ambiente)
├── evaluation/          # Métricas e utilitários de ranking
├── models/              # Recomendadores, Factory, Registry e inferência
│   ├── inference.py      # RecommendationEngine (predição)
│   ├── model_loader.py  # ModelFactory (Factory Pattern)
│   ├── registry_recommender.py  # Carrega do MLflow Model Registry
│   ├── item_based_cf.py # Modelo Production
│   ├── popularity.py    # Fallback cold-start
│   ├── ncf.py           # Rede neural (PyTorch)
│   └── training/        # Rotinas de treino dos 5 modelos (portadas dos notebooks)
├── pipeline/            # Pipeline DVC (preprocess -> train -> evaluate)
│   ├── common.py        # Seed, config, dataset_hash, MLflow setup
│   ├── preprocess.py    # Estágio 1: split StratifiedKFold, vocabulários, matriz esparsa
│   ├── train.py         # Estágio 2: treino dos 5 modelos
│   └── evaluate.py      # Estágio 3: comparação, MODEL_CARD, Model Registry
├── preprocessing/       # Strategy Pattern (InteractionFilter, UserItemEncoder)
└── serving/             # MLflow pyfunc wrapper para o Registry
tests/                    # Testes unitários (pytest)
notebooks/                # EDA → pré-processamento → modelos → comparação
configs/                  # Hiperparâmetros (YAML)
data/                     # raw/ (DVC) e processed/ (saída do pipeline)
models/                   # Artefatos, métricas e MODEL_CARD.md
scripts/                  # validate_env.py, gen_synthetic_data.py
docs/                     # Documentação de convenções e requisitos
dvc.yaml                  # Pipeline DVC (3 estágios)
dvc.lock                  # Lock do pipeline
Dockerfile                # Multi-stage (builder + runtime)
docker-compose.yml        # Orquestração (API + MLflow)
.github/workflows/ci.yml  # CI: ruff + format + lock + pytest
```

---

## Stack Tecnológico

| Componente | Tecnologia |
|-----------|----------|
| Linguagem | Python 3.12+ |
| Modelagem | PyTorch + Scikit-Learn |
| API | FastAPI + Uvicorn |
| Pipeline | DVC (3 estágios, remote local) |
| Rastreamento | MLflow Tracking + Model Registry (Staging → Production) |
| Container | Docker multi-stage + docker-compose |
| CI | GitHub Actions (ruff, format, lock, pytest) |
| Gerenciador de Deps | uv + pyproject.toml |
| Linting | Ruff |
| Logging | Structlog |
| Testes | pytest |

---

## Design Patterns

### Factory Pattern
`ModelFactory` em [`src/models/model_loader.py`](src/models/model_loader.py) registra
e instancia os recomendadores (`item_based_cf`, `item_based_cf_registry`, `popularity`)
sem que o código cliente conheça a classe concreta.

### Strategy Pattern
`src/preprocessing/` implementa `InteractionFilterStrategy` e
`UserItemEncoderStrategy`, encapsulando algoritmos de pré-processamento
intercambiáveis.

### Dependency Injection
`src/api/dependencies.py` injeta o `RecommendationEngine` nas rotas via `Depends`,
com carregamento único (singleton) e erro `503` caso os artefatos não possam
ser carregados.

---

## Resultados

Os artefatos commitados (`models/`, `data/processed/`) foram gerados com um
**dataset sintético** para validação rápida do pipeline. Para reproduzir com o
dataset real do Kaggle, coloque os CSVs em `data/raw/` e rode `uv run dvc repro`.

### Dataset real (Kaggle)

Avaliação no split de teste interno (15% dos usuários; catálogo restrito aos
3.000 produtos mais comprados). Resultados completos em
[`models/MODEL_CARD.md`](models/MODEL_CARD.md).

O **Item-based CF** liderou em `Recall@10` e `NDCG@10` com alta cobertura e
baixa latência, sendo promovido a `Production`. A rede neural **NCF**
(modelo principal em PyTorch) foi implementada com embeddings + MLP, random
search e early stopping, mas não superou os baselines de CF neste dataset —
análise da causa-raiz em [`docs/NOTEBOOKS.md`](docs/NOTEBOOKS.md) (seção 7.3).

### Dataset sintético (commitado)

Com o dataset sintético, o baseline de **popularidade** empata com Item-based
CF em `Recall@10` (ambos ~0,50) e vence em `NDCG@10` por ser mais simples — o
comportamento é esperado: dados sintéticos não reproduzem o viés de recompra
do Instacart real, onde Item-based CF se destaca.

---

## Autor

Desenvolvido para FIAP - Tech Challenge 2

## Licença

MIT
