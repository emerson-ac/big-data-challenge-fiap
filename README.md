# Sistema de Recomendação de Produtos - Tech Challenge 2

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)

## Visão Geral

Sistema de recomendação de produtos para e-commerce baseado em comportamento de navegação e compra de usuários. Implementa uma rede neural (MLP com embeddings) em PyTorch, comparada com baselines de Scikit-Learn, rastreamento via MLflow e versionamento de dados com DVC.

**Dataset:** [Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset) (~50 milhões de interações, +3 milhões de usuários)

---

## Requisitos Obrigatórios

✅ **Estrutura:** Diretórios `src/`, `tests/`, `data/`, `models/`, `configs/`  
✅ **Código:** Funções ≤ 20 linhas, type hints, docstrings Google Style  
✅ **Padrões:** Implementação de Design Patterns (Factory, Strategy)  
✅ **Ambiente:** Gerenciamento via `pyproject.toml` e `uv`  
✅ **ML:** Modelo PyTorch + Baselines Scikit-Learn, 4+ métricas  
✅ **MLOps:** Docker, DVC (3+ estágios), MLflow Tracking + Registry  
✅ **Qualidade:** Ruff linting, pre-commit hooks, commits semânticos  

Ver [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) para lista completa.

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
