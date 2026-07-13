# Sistema de Recomendação de Produtos - Tech Challenge 2

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)

## Visão Geral

Sistema de recomendação de produtos para e-commerce baseado em comportamento de navegação e compra de usuários. Implementa uma rede neural (MLP com embeddings) em PyTorch, comparada com baselines de Scikit-Learn, rastreamento via MLflow e versionamento de dados com DVC.

**Dataset:** [Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset) (~3,4 milhões de pedidos, ~206 mil usuários, ~34 milhões de interações pedido-produto)

---

## Requisitos Obrigatórios

- [x] **Estrutura:** Diretórios `src/`, `tests/`, `data/`, `models/`, `configs/`
- [x] **Código:** Funções ≤ 20 linhas, type hints, docstrings Google Style
- [x] **Padrões:** Design Patterns — Factory (`src/models/model_loader.py`) e Strategy (`src/api/services/enrichment.py`)
- [x] **Ambiente:** Dependências de prod/dev (pytorch, sklearn, mlflow, dvc) via `pyproject.toml` e `uv`; config externalizada em `.env` (Pydantic Settings)
- [x] **ML:** Rede neural PyTorch (NCF) + 4 baselines Scikit-Learn, 6 métricas de ranking
- [x] **MLOps:** Docker multi-stage + docker-compose, DVC (3 estágios: preprocess → train → evaluate), MLflow Tracking + Model Registry
- [x] **Qualidade:** Ruff sem erros, pre-commit hooks, Conventional Commits

Ver [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) para o checklist completo e atualizado.

---

## Estrutura do Projeto

```
.
├── src/
│   ├── api/               # API REST (FastAPI) + Strategy Pattern (services/)
│   ├── pipeline/          # Estágios DVC: preprocess, train, evaluate
│   ├── models/            # Modelos, Factory (model_loader) e rotinas de treino
│   ├── evaluation/        # Métricas de ranking e utilitários
│   ├── serving/           # Wrapper pyfunc p/ o MLflow Registry
│   └── config.py          # Settings externalizadas (Pydantic + .env)
├── tests/                 # Testes unitários (pytest)
├── notebooks/             # 01_eda → 02_preprocessing → 03..07 modelos → 08_comparison
├── data/
│   ├── raw/               # Dados brutos (Kaggle, não commitados)
│   └── processed/         # Dados processados (gerados pelo pipeline)
├── models/                # Artefatos de modelos + MODEL_CARD.md
│   ├── sklearn-iris/       # BÔNUS: modelo demo KServe
│   └── recsys/             # BÔNUS: predictor KServe do recomendador
├── configs/               # Configurações YAML
├── scripts/               # validate_env, gen_synthetic_data, deploy/bootstrap (bônus)
├── cluster/ · platform/   # BÔNUS: infra KServe/EKS declarativa
├── docs/                  # Documentação (REQUIREMENTS, NOTEBOOKS, patterns...)
├── Dockerfile             # Containerização multi-stage
├── docker-compose.yml     # API + servidor MLflow
├── dvc.yaml / dvc.lock    # Pipeline reprodutível (3 estágios)
├── .env.example           # Modelo de configuração
├── pyproject.toml         # Dependências e configs
└── README.md              # Este arquivo
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

# Configurar variáveis de ambiente
cp .env.example .env

# Validar ambiente (Python, dependências e estrutura)
uv run python scripts/validate_env.py
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

O pipeline reprodutível vive em `src/pipeline/` e é orquestrado pelo DVC
(3 estágios). A lógica é a mesma dos notebooks `02`–`08`, portada para scripts.

```bash
# Pipeline completo e reprodutível (preprocess → train → evaluate)
uv run dvc repro

# Ou estágio a estágio:
uv run python -m src.pipeline.preprocess   # data/processed/*
uv run python -m src.pipeline.train        # treina os 5 modelos → models/*
uv run python -m src.pipeline.evaluate     # comparação + MODEL_CARD + Registry
```

> **Sem os dados do Kaggle?** Gere um mini-dataset sintético para exercitar o
> pipeline ponta a ponta em segundos:
> `uv run python scripts/gen_synthetic_data.py && uv run dvc repro`.
> O `dvc.lock` versionado foi gerado nessa validação sintética; rode `dvc repro`
> com os CSVs reais em `data/raw/` para regenerar os artefatos e o lock reais.

Os notebooks (`notebooks/01_eda.ipynb` … `08_model_comparison.ipynb`) permanecem
como registro exploratório de cada modelo.

### 4. Rastrear Experimentos

O tracking usa um **servidor MLflow externo** por padrão
(`RECSYS_MLFLOW_TRACKING_URI=https://mlflow.pocsarcotech.com`, MLflow 3.x). Os
experimentos são namespaced sob `recsys-instacart/*` e o melhor modelo é promovido
por **alias `@production`** no Model Registry (MLflow 3 removeu os *stages*). Os
artefatos são proxiados pelo servidor — **não** são necessárias credenciais AWS.

```bash
# UI do servidor externo
open https://mlflow.pocsarcotech.com

# Dev offline: backend sqlite (o MLflow 3 descontinuou o file store)
export RECSYS_MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # http://localhost:5000
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

## CI/CD (GitHub Actions)

Três workflows automatizam qualidade, retreino e deploy (a infra KServe do bônus
continua no `deploy.yml`/`provision-cluster.yml`, intactos):

| Workflow | Gatilho | O que faz |
|---|---|---|
| `ci.yml` | Pull Request | `ruff check src/ tests/` + `pytest` (offline) |
| `model-release.yml` | Push na `main` (pipeline/modelos) · manual | `dvc pull` (S3) → `dvc repro` → **atualiza o modelo no MLflow** (alias `@production`) → `dvc push` |
| `deploy-api.yml` | Push na `main` (api/Dockerfile/k8s) · manual | build+push da imagem (Docker Hub) → `kubectl apply -k k8s/api` no EKS → rollout + rollback |

### Dados de treino no CI (remote DVC em S3)

O `model-release` treina no dado real, versionado por DVC em S3
(`s3://arcobridgegitops-models-*/dvc`). Configuração **única** na sua máquina:

```bash
# 1. Baixe os CSVs do Instacart (Kaggle) para data/raw/
# 2. Publique dados + artefatos no S3 e atualize o dvc.lock
bash scripts/setup_dvc_s3.sh
git add dvc.lock && git commit -m "chore: publicar dados no remote S3"
```

### Secrets / variables necessários

| Nome | Tipo | Uso |
|---|---|---|
| `AWS_ROLE_ARN` | variable | OIDC para EKS e S3 (já usado pelo `deploy.yml`) |
| `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` | secret | Push da imagem da API |
| `RECSYS_MLFLOW_TRACKING_URI` | secret (opcional) | Sobrescreve o servidor MLflow (default: externo) |

---

## Arquitetura e Design Patterns

### Factory Pattern
Usado em `src/models/model_loader.py` (`ModelFactory`) para registrar e instanciar
diferentes recomendadores pelo nome, desacoplando o engine do tipo de modelo.

### Strategy Pattern
Usado em `src/api/services/enrichment.py` para alternar estratégias de
enriquecimento da resposta (`IdOnlyStrategy` vs `WithNamesStrategy`), selecionadas
em tempo de requisição pela flag `enrich_names`.

### Dependency Injection
FastAPI `Depends` injeta o engine e o mapa de nomes nos endpoints
(`src/api/dependencies.py`).

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

O **Item-based CF** foi promovido via alias `@production` no MLflow Model Registry por obter o melhor `Recall@10` e `NDCG@10` entre os 5 modelos — as métricas mais relevantes para o problema (cobertura do que o usuário de fato recompra e qualidade do ranking), além de aliar o segundo melhor `Coverage@10` (80,73% do catálogo é recomendado em algum momento, evitando recomendar sempre os mesmos itens populares) com baixíssima latência de inferência (0,08 ms/usuário) e tamanho de modelo modesto (29 MB).

A rede neural (**NCF**, modelo principal exigido em PyTorch pelo Tech Challenge) foi implementada com embeddings de usuário/item + MLP, random search de hiperparâmetros e early stopping, mas **não superou os baselines de Collaborative Filtering** neste dataset — ficou no mesmo nível do baseline ingênuo de popularidade. A causa raiz está documentada em [`docs/NOTEBOOKS.md`](docs/NOTEBOOKS.md) (seção 7.3): o Instacart tem um viés de recompra muito forte (os mesmos produtos voltam a ser comprados repetidamente), um padrão que a similaridade de co-ocorrência item-a-item captura diretamente, enquanto a rede neural precisa aprendê-lo a partir de amostragem negativa e poucos dados — exigindo mais dados, épocas ou capacidade do modelo para superar técnicas clássicas mais simples nesse cenário. Por isso, apenas o Item-based CF recebe o alias `@production`: a NCF foi implementada, comparada e documentada conforme exigido, mas não promovida por desempenho inferior aos baselines.

---

## Bônus — Opção 2: Deploy em Kubernetes com KServe/EKS

Além do serving principal (FastAPI + Docker acima), o projeto inclui, **como
segunda opção / bônus**, uma infraestrutura declarativa para servir o modelo em
um cluster **Kubernetes (EKS)** via **KServe**, com CI/CD por GitHub Actions
(OIDC), Ingress ALB e certificado TLS.

| Componente | Onde |
|---|---|
| Cluster EKS declarativo (eksctl) | `cluster/cluster.yaml` |
| Plataforma (cert-manager, AWS LB Controller, KServe) | `platform/` + `scripts/deploy.sh` |
| Modelo demo (sklearn/Iris, formato nativo) | `models/sklearn-iris/` |
| **Predictor do recomendador (custom container)** | `models/recsys/` |
| CI/CD (provision, deploy, destroy) | `.github/workflows/` |

O predictor em `models/recsys/` empacota o **mesmo `RecommendationEngine`** da API
(modelo Production, item-based CF) como um *custom container* KServe — ver
[`models/recsys/README.md`](models/recsys/README.md).

> **Observações:** (1) é um caminho **alternativo** ao exigido (Docker/DVC/MLflow),
> não um substituto. (2) Os valores de AWS (conta, VPC, bucket, domínio) são
> parametrizáveis por ambiente (`.env` / variáveis do CI), com os defaults da PoC
> original. (3) Requer uma conta AWS e um cluster para ser exercitado de verdade.

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
