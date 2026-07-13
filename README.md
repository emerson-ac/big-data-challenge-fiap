# Sistema de Recomendação de Produtos - Tech Challenge 2

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138+-green)

## Visão Geral

Sistema de recomendação de produtos para e-commerce baseado em comportamento de compra de usuários. Compara 5 modelos (popularidade, item-based CF, user-based CF, matrix factorization e uma rede neural NCF em PyTorch), com experimentos rastreados no MLflow, e expõe o modelo Production via uma API REST (FastAPI).

**Dataset:** [Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset). Os arquivos brutos e processados já estão incluídos em `data/` — não é necessário baixar nada para rodar o projeto.

---

## Status do Projeto

Ver [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) para o checklist completo.

- [x] **Estrutura:** Diretórios `src/`, `tests/`, `data/`, `models/`, `configs/`
- [x] **Código:** Funções ≤ 20 linhas, type hints, docstrings Google Style
- [x] **Padrões:** Factory Pattern (`src/models/model_loader.py`) + Dependency Injection (FastAPI)
- [x] **Ambiente:** `pyproject.toml` + `uv`, lock file commitado, deps prod/dev separadas
- [x] **ML:** Baselines Scikit-Learn + rede neural PyTorch (NCF), 8 métricas de avaliação
- [x] **MLflow:** Tracking de múltiplos runs + Model Registry (`item_based_cf_recommender` em Production)
- [x] **API:** FastAPI servindo o modelo Production (`src/api/`)
- [x] **Qualidade:** Ruff sem erros, pre-commit hooks, commits semânticos
- [ ] **Docker:** Dockerfile/docker-compose ainda não implementados
- [ ] **DVC:** Dados versionados diretamente no Git por ora; pipeline `dvc.yaml` ainda não implementado

---

## Estrutura do Projeto

```
.
├── src/
│   ├── api/                # API REST (FastAPI)
│   │   ├── main.py         # Aplicação FastAPI (lifespan, exception handlers, rotas)
│   │   ├── config.py       # Settings via Pydantic (.env)
│   │   ├── dependencies.py # Injeção de dependências
│   │   ├── routes/         # health, recommendations
│   │   ├── schemas/        # request, response, errors (Pydantic)
│   │   ├── services/       # recommendation_service.py
│   │   ├── middleware/     # error_handler.py
│   │   └── utils/          # logger.py (structlog)
│   ├── models/              # Modelos e camada de inferência
│   │   ├── inference.py     # RecommendationEngine (ponto de entrada de predição)
│   │   ├── model_loader.py  # ModelFactory (Factory Pattern)
│   │   ├── item_based_cf.py # Modelo Production
│   │   ├── popularity.py    # Fallback de cold-start
│   │   ├── ncf.py           # Rede neural (PyTorch)
│   │   └── recommender.py   # Protocol comum
│   └── evaluation/          # metrics.py, ranking.py (compartilhados pelos notebooks)
├── tests/                    # Testes unitários e de integração (pytest)
├── notebooks/                # Pipeline de modelagem, numerado por ordem de execução
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_popularity.ipynb
│   ├── 04_item_based_cf.ipynb
│   ├── 05_user_based_cf.ipynb
│   ├── 06_matrix_factorization.ipynb
│   ├── 07_ncf_training.ipynb
│   └── 08_model_comparison.ipynb
├── data/
│   ├── raw/                 # Dados brutos do Instacart (já incluídos)
│   └── processed/           # Dados processados/splits (já incluídos)
├── models/                   # Artefatos treinados + MODEL_CARD.md
├── configs/
│   └── model_config.yaml     # Hiperparâmetros e seeds de cada modelo
├── mlruns/                    # Tracking store local do MLflow
├── docs/                      # Documentação de convenções e requisitos
├── .env.example               # Variáveis de ambiente da API
├── pyproject.toml              # Dependências e configs (uv)
└── README.md
```

---

## Stack Tecnológico

| Componente | Tecnologia |
|-----------|----------|
| **Linguagem** | Python 3.12+ |
| **Modelagem** | PyTorch + Scikit-Learn |
| **API** | FastAPI + Uvicorn |
| **Rastreamento** | MLflow Tracking + Model Registry |
| **Gerenciador de Deps** | uv + pyproject.toml |
| **Linting** | Ruff |
| **Logging** | Structlog |
| **Testes** | pytest |

---

## Quick Start

### 1. Configurar Ambiente

```bash
# Clonar repositório
git clone https://github.com/emerson-ac/big-data-challenge-fiap.git
cd big-data-challenge-fiap

# Instalar dependências (cria .venv automaticamente)
uv sync

# Ativar git hooks de qualidade (ruff, uv lock check)
uv run pre-commit install
```

### 2. Dados

Os dados brutos (`data/raw/*.csv`) e processados (`data/processed/*`) já estão incluídos no repositório — nenhum download é necessário. Caso queira reprocessar a partir da fonte original, o dataset está disponível em [Kaggle](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset).

### 3. (Opcional) Reexecutar o Pipeline de Modelagem

Os notebooks já foram executados e seus artefatos (`models/`, `mlruns/`) estão commitados — **não é necessário rodá-los para usar a API**. Para reproduzir o treinamento do zero:

```bash
uv run jupyter notebook notebooks/
# Executar em ordem: 01_eda -> 02_preprocessing -> 03_baseline_popularity ->
# 04_item_based_cf -> 05_user_based_cf -> 06_matrix_factorization ->
# 07_ncf_training -> 08_model_comparison
```

### 4. (Opcional) Inspecionar Experimentos no MLflow

```bash
uv run mlflow ui --host 0.0.0.0 --port 5000
# Acessar: http://localhost:5000
```

### 5. Iniciar a API

```bash
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Swagger UI: http://localhost:8000/docs
# Health check: http://localhost:8000/health/status
```

Exemplo de requisição:

```bash
curl -X POST http://localhost:8000/recommendations/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 5}'
```

A API carrega o modelo `item_based_cf` (Production no MLflow Model Registry) e cai automaticamente no fallback de popularidade para usuários sem histórico conhecido (cold-start). Configurações (caminhos de artefatos, `top_k` padrão, etc.) são externalizadas via `.env` — ver [.env.example](.env.example).

---

## Desenvolvimento

### Linting e Formatação

```bash
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .
```

### Testes

```bash
# Executar testes
uv run pytest tests/ -v

# Com cobertura
uv run pytest tests/ --cov=src
```

### Commits

Utilizar padrão semântico (ver [docs/COMMIT-CONVENTIONS.md](docs/COMMIT-CONVENTIONS.md)):
```bash
git commit -m "feat: implementar modelo neural"
git commit -m "fix: corrigir normalização de features"
git commit -m "refactor: extrair service de recomendação"
```

---

## Arquitetura e Design Patterns

### Factory Pattern
`ModelFactory` em [`src/models/model_loader.py`](src/models/model_loader.py) registra e instancia os recomendadores (`item_based_cf`, `popularity`) sem que o código cliente conheça a classe concreta — novos modelos são adicionados via `ModelFactory.register(...)`.

### Dependency Injection
`src/api/dependencies.py` injeta o `RecommendationEngine`/`RecommendationService` nas rotas via `Depends`, com carregamento único (singleton) e erro `503` (`ModelNotLoadedError`) caso os artefatos do modelo não possam ser carregados.

---

## Métricas de Avaliação

- **Precision@K** - Proporção de itens relevantes no top-K
- **Recall@K** - Cobertura de itens relevantes
- **NDCG (Normalized Discounted Cumulative Gain)** - Qualidade do ranking
- **MAP (Mean Average Precision)** - Média de precisão em rankings

---

## Resultados dos Modelos

Os 5 modelos foram treinados e avaliados no mesmo split de teste interno (15% dos usuários, catálogo restrito aos 3.000 produtos mais comprados — 73,1% do volume de compras). Resultados completos em [`models/MODEL_CARD.md`](models/MODEL_CARD.md) e [`models/evaluation/metrics_comparison.csv`](models/evaluation/metrics_comparison.csv), gerados pelo notebook [`08_model_comparison.ipynb`](notebooks/08_model_comparison.ipynb).

| Modelo | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Hit Rate@10 | Coverage@10 | Latência (ms/usuário) |
|---|---|---|---|---|---|---|---|
| Popularidade (baseline) | 0,0747 | 0,0935 | 0,1097 | 0,0522 | 0,4729 | 0,33% | 0,0001 |
| **Item-based CF** | 0,1292 | **0,2236** | **0,2172** | **0,1274** | 0,6722 | 80,73% | 0,0815 |
| User-based CF (KNN) | 0,1270 | 0,2051 | 0,2117 | 0,1227 | 0,6657 | 72,97% | 0,5362 |
| Matrix Factorization (SVD) | **0,1398** | 0,2017 | 0,2032 | 0,1137 | 0,6636 | 21,53% | 0,0344 |
| NCF (rede neural, PyTorch) | 0,0750 | 0,0930 | 0,1093 | 0,0519 | 0,4726 | 1,43% | 4,9801 |

### Por que o Item-based CF foi escolhido como melhor modelo

O **Item-based CF** foi promovido a `Production` no MLflow Model Registry por obter o melhor `Recall@10` e `NDCG@10` entre os 5 modelos — as métricas mais relevantes para o problema (cobertura do que o usuário de fato recompra e qualidade do ranking), além de aliar o segundo melhor `Coverage@10` (80,73% do catálogo é recomendado em algum momento, evitando recomendar sempre os mesmos itens populares) com baixíssima latência de inferência (0,08 ms/usuário) e tamanho de modelo modesto (29 MB). É esse o modelo servido pela API (`src/api/`).

A rede neural (**NCF**, modelo principal exigido em PyTorch pelo Tech Challenge) foi implementada com embeddings de usuário/item + MLP, random search de hiperparâmetros e early stopping, mas **não superou os baselines de Collaborative Filtering** neste dataset — ficou no mesmo nível do baseline ingênuo de popularidade. A causa raiz está documentada em [`docs/NOTEBOOKS.md`](docs/NOTEBOOKS.md) (seção 7.3): o Instacart tem um viés de recompra muito forte (os mesmos produtos voltam a ser comprados repetidamente), um padrão que a similaridade de co-ocorrência item-a-item captura diretamente, enquanto a rede neural precisa aprendê-lo a partir de amostragem negativa e poucos dados — exigindo mais dados, épocas ou capacidade do modelo para superar técnicas clássicas mais simples nesse cenário. Por isso, o `ncf_recommender` permanece registrado em `Staging`: implementado, comparado e documentado conforme exigido, mas não promovido por desempenho inferior aos baselines.

---

## Contribuindo

1. Criar branch: `git checkout -b feature/minha-feature`
2. Cometer mudanças: `git commit -m "feat: descrição"`
3. Push: `git push origin feature/minha-feature`
4. Abrir Pull Request

---

## Autor

Desenvolvido para FIAP - Tech Challenge 2

---

## Licença

MIT
