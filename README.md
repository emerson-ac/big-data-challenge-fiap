# Sistema de Recomendação de Produtos - Tech Challenge 2

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)

## Visão Geral

Sistema de recomendação de produtos para e-commerce baseado em comportamento de navegação e compra de usuários. Implementa uma rede neural (MLP com embeddings) em PyTorch, comparada com baselines de Scikit-Learn, rastreamento via MLflow e versionamento de dados com DVC.

**Dataset:** [Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset) (~3,4 milhões de pedidos, ~206 mil usuários, ~34 milhões de interações pedido-produto)

---

## Requisitos Obrigatórios

Status atual: **Etapa 1 (Clean Code e Estrutura)** em andamento.

- [x] **Estrutura:** Diretórios `src/`, `tests/`, `data/`, `models/`, `configs/`
- [ ] **Código:** Funções ≤ 20 linhas, type hints, docstrings Google Style
- [ ] **Padrões:** Implementação de Design Patterns (Factory, Strategy)
- [ ] **Ambiente:** Dependências de prod/dev (pytorch, sklearn, mlflow, dvc) via `pyproject.toml` e `uv`
- [ ] **ML:** Modelo PyTorch + Baselines Scikit-Learn, 4+ métricas
- [ ] **MLOps:** Docker, DVC (3+ estágios), MLflow Tracking + Registry
- [ ] **Qualidade:** Ruff linting, pre-commit hooks, commits semânticos

Ver [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) para o checklist completo e atualizado.

---

## Estrutura do Projeto

```
.
├── src/                    # Código principal
│   ├── api/               # API REST (FastAPI)
│   ├── models/            # Modelos de ML
│   └── utils/             # Utilitários
├── tests/                 # Testes unitários
├── notebooks/             # Análise e experimentos
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
├── data/
│   ├── raw/              # Dados brutos (não commitados)
│   └── processed/        # Dados processados (versionado via DVC)
├── models/               # Artefatos de modelos
├── configs/              # Configurações YAML
├── docs/                 # Documentação
│   ├── REQUIREMENTS.md
│   ├── NOTEBOOKS.md
│   ├── api-conventions.md
│   ├── naming-conventions.md
│   └── design-pattern.md
├── Dockerfile            # Containerização
├── docker-compose.yml
├── dvc.yaml             # Pipeline DVC
├── pyproject.toml       # Dependências e configs
└── README.md            # Este arquivo
```

---

## Stack Tecnológico

| Componente | Tecnologia |
|-----------|----------|
| **Linguagem** | Python 3.11+ |
| **Modelagem** | PyTorch + Scikit-Learn |
| **API** | FastAPI + Uvicorn |
| **Rastreamento** | MLflow Tracking + Model Registry |
| **Versionamento de Dados** | DVC |
| **Containerização** | Docker + Docker Compose |
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

# Instalar dependências e ativar git hooks
uv sync
uv run pre-commit install

# Validar ambiente
uvicorn python scripts/validate_env.py
```

### 2. Preparar Dados

```bash
# Baixar dataset Instacart
# https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset
# Extrair para: data/raw/

ls data/raw/
# aisles.csv
# departments.csv
# order_products__prior.csv
# order_products__train.csv
# orders.csv
# products.csv
```

### 3. Executar Pipeline

```bash
# Usando DVC (reprodutível)
dvc repro

# Ou manualmente:
# 1. EDA
jupyter notebook notebooks/01_eda.ipynb

# 2. Baselines
jupyter notebook notebooks/02_baseline.ipynb

# 3. Pré-processamento
jupyter notebook notebooks/03_preprocessing.ipynb

# 4. Treinamento
jupyter notebook notebooks/04_model_training.ipynb

# 5. Avaliação
jupyter notebook notebooks/05_evaluation.ipynb
```

### 4. Rastrear Experimentos

```bash
# Iniciar MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Acessar: http://localhost:5000
```

### 5. Iniciar API

```bash
# Desenvolvimento
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Produção (Docker)
docker-compose up

# Acessar: http://localhost:8000/docs
```

---

## Desenvolvimento

### Linting e Formatação

```bash
# Verificar qualidade de código
uv run ruff check .

# Corrigir automaticamente
uv run ruff check . --fix
```

### Testes

```bash
# Executar testes
uv run pytest tests/ -v

# Com cobertura
uv run pytest tests/ --cov=src/
```

### Commits

Utilizar padrão semântico:
```bash
git commit -m "feat: implementar modelo neural"
git commit -m "fix: corrigir normalização de features"
git commit -m "refactor: extrair service de recomendação"
```

---

## Arquitetura e Design Patterns

### Factory Pattern
Usado em `src/models/model_loader.py` para instanciar diferentes tipos de modelos.

### Strategy Pattern
Usado em `src/api/services/preprocessing_service.py` para diferentes estratégias de pré-processamento.

### Dependency Injection
FastAPI dependencies para injeção de services em endpoints.

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

O **Item-based CF** foi promovido a `Production` no MLflow Model Registry por obter o melhor `Recall@10` e `NDCG@10` entre os 5 modelos — as métricas mais relevantes para o problema (cobertura do que o usuário de fato recompra e qualidade do ranking), além de aliar o segundo melhor `Coverage@10` (80,73% do catálogo é recomendado em algum momento, evitando recomendar sempre os mesmos itens populares) com baixíssima latência de inferência (0,08 ms/usuário) e tamanho de modelo modesto (29 MB).

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
