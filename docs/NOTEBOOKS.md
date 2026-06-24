# Notebooks do Projeto

Este arquivo documenta a estrutura, propósito e fluxo de execução dos notebooks utilizados no projeto de sistema de recomendação.

---

## Visão Geral

Os notebooks são organizados por fase do projeto e devem ser executados em ordem específica para garantir reprodutibilidade e rastreabilidade das análises. Cada modelo (baseline e principal) tem seu próprio notebook, todos consumindo os mesmos splits/vocabulários gerados pelo `02_preprocessing.ipynb`, para garantir comparação justa no `08_model_comparison.ipynb`.

**Estrutura de Diretórios:**
```
notebooks/
├── 01_eda.ipynb                   # Análise exploratória de dados
├── 02_preprocessing.ipynb         # Split treino/val/teste, vocabulários, matriz esparsa
├── 03_baseline_popularity.ipynb   # Baseline: popularidade global
├── 04_item_based_cf.ipynb         # Collaborative Filtering item-based
├── 05_user_based_cf.ipynb         # Collaborative Filtering user-based (KNN)
├── 06_matrix_factorization.ipynb  # Matrix Factorization implícita (TruncatedSVD)
├── 07_ncf_training.ipynb          # Modelo principal: Neural Collaborative Filtering (PyTorch)
└── 08_model_comparison.ipynb      # Avaliação e comparação final de todos os modelos
```

### Design de Split Treino/Validação/Teste

O dataset Instacart tem uma particularidade importante: a coluna `eval_set` em `orders.csv` marca cada usuário como `prior`, `train` ou `test`, mas **apenas os usuários `train` têm o "cesto futuro" revelado** em `order_products__train.csv` (os usuários `test` eram a submissão oculta da competição original no Kaggle — não temos o gabarito).

Por isso:
- **`prior`** (todos os usuários): histórico de compras, usado como contexto/input para todos os modelos (popularidade, similaridade, embeddings).
- **`train`** (~131k usuários, único grupo com rótulo conhecido): o "cesto futuro" de cada usuário nesse grupo é o rótulo positivo. Esses usuários são divididos em **nosso** treino/validação/teste internos (70/15/15, estratificado pelo segmento de usuário definido na EDA: `ocasional`/`regular`/`super_user`).
- **`test`** (eval_set original do Kaggle): descartado — sem gabarito, não é utilizável para treinar nem avaliar.

Essa lógica é implementada uma única vez em `02_preprocessing.ipynb` e reaproveitada por todos os notebooks de modelo.

### Escopo de Catálogo (Top-N Produtos)

Para viabilizar similaridade item-item/usuário-usuário e treino do NCF em tempo de execução razoável, o catálogo de avaliação é restrito aos **top 3.000 produtos mais comprados** (cobrem a grande maioria do volume de compras, dado o efeito de cauda longa identificado na EDA). Essa decisão é tomada uma única vez em `02_preprocessing.ipynb`, documentada no `data_profile.json`/log do MLflow, e aplicada de forma consistente a **todos** os modelos — não é um corte silencioso por notebook.

---

## 1. EDA - Exploratory Data Analysis (`01_eda.ipynb`)

**Propósito:** Entender a estrutura, distribuições e características dos dados brutos.

**Inputs:**
- `data/raw/aisles.csv`
- `data/raw/departments.csv`
- `data/raw/products.csv`
- `data/raw/orders.csv`
- `data/raw/order_products__prior.csv`
- `data/raw/order_products__train.csv`

**Outputs:**
- Visualizações estatísticas (salvos em `notebooks/outputs/eda/`)
- Relatório de dados ausentes, outliers e distribuições
- Insights sobre padrões de compra e comportamento do usuário
- Arquivo de metadados: `data/processed/data_profile.json`

**Etapas Principais:**
1. Carregamento e verificação inicial dos dados
2. Análise de valores ausentes, duplicatas e tipos de dados
3. Estatísticas descritivas por tabela
4. Análise de distribuições de compra (frequência, valor)
5. Segmentação de usuários (super-users, ocasionais, inativos)
6. Análise de popularidade de produtos e departamentos
7. Correlações e padrões de co-compra
8. Visualizações: histogramas, boxplots, heatmaps

**Dependências:**
- pandas, numpy
- matplotlib, seaborn
- scipy (para testes estatísticos)

**Rastreamento MLflow:** Não aplicável (exploração somente)

---

## 2. Preprocessing - Splits e Vocabulários (`02_preprocessing.ipynb`)

**Propósito:** Gerar, uma única vez, os splits treino/validação/teste, os vocabulários de usuário/item e a matriz esparsa de interações usadas por **todos** os notebooks de modelo (seções 3 a 7).

**Inputs:**
- `data/raw/orders.csv`, `data/raw/order_products__prior.csv`, `data/raw/order_products__train.csv`, `data/raw/products.csv`
- `data/processed/data_profile.json` (segmentação de usuários gerada na EDA)

**Outputs:**
- `data/processed/vocabularies.pkl` — mapeamentos `user_id ↔ user_idx`, `product_id ↔ product_idx` (restrito ao top-N catálogo)
- `data/processed/interactions_prior.npz` — matriz esparsa `(n_users, n_items)` binária, histórico de compras (`prior`)
- `data/processed/train_pairs.pkl`, `data/processed/val_pairs.pkl`, `data/processed/test_pairs.pkl` — pares positivos `(user_idx, item_idx)` por split, derivados do cesto `train` do Instacart
- `data/processed/split_meta.json` — proporções, contagem de usuários por split, hash do dataset, top-N do catálogo

**Etapas Principais:**
1. Filtrar `orders.eval_set == "prior"` e construir a matriz esparsa de histórico de interações
2. Filtrar `orders.eval_set == "train"` e extrair o cesto-rótulo de cada usuário via `order_products__train.csv`
3. Selecionar o top-N catálogo de produtos (`N = 3000`) por volume de compras
4. Construir vocabulários `user_id ↔ idx` e `product_id ↔ idx` restritos a esse catálogo
5. Split estratificado dos usuários `train` em treino/validação/teste internos (70/15/15) por segmento de usuário (`StratifiedKFold`/`train_test_split(stratify=...)`)
6. Serializar vocabulários, matriz esparsa e pares por split
7. Registrar hash do dataset (SHA256) e metadados do split

**Dependências:**
- pandas, numpy, scipy.sparse
- scikit-learn (`train_test_split` estratificado)
- pickle, hashlib

**Rastreamento MLflow:**
- Log de params: `top_n_products`, `train_ratio`, `val_ratio`, `test_ratio`, `random_seed`
- Log de métricas: `n_users_train`, `n_users_val`, `n_users_test`, `n_items_catalog`
- Log de artefatos: `split_meta.json`

---

## 3. Baseline - Popularidade (`03_baseline_popularity.ipynb`)

**Propósito:** Modelo de referência mais simples — recomenda os produtos mais comprados globalmente, ignorando qualquer personalização. É o piso de comparação: qualquer modelo mais sofisticado deve superá-lo.

**Inputs:**
- `data/processed/vocabularies.pkl`, `data/processed/interactions_prior.npz`
- `data/processed/{train,val,test}_pairs.pkl`

**Outputs:**
- `models/baseline_popularity/ranking.pkl` — ranking global de produtos por frequência
- `models/baseline_popularity/metrics.json`

**Etapas Principais:**
1. Carregar vocabulário e matriz de interações `prior`
2. Calcular frequência de compra por produto (soma da matriz esparsa por coluna)
3. Gerar ranking global fixo (top-K igual para todos os usuários)
4. Avaliar nos splits de validação e teste com as métricas oficiais (seção 9.1)
5. Persistir ranking e métricas

**Dependências:**
- numpy, scipy.sparse
- pandas

**Rastreamento MLflow:**
- Log de params: `k` (tamanho do top-K avaliado)
- Log de métricas: `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `map_at_k` (val e teste)
- Log de artefatos: `ranking.pkl`, `metrics.json`

---

## 4. Collaborative Filtering Item-Based (`04_item_based_cf.ipynb`)

**Propósito:** Recomendar produtos similares aos que o usuário já comprou, via similaridade de co-ocorrência entre itens. Captura o padrão dominante do Instacart ("quem compra X também compra Y") com baixo custo computacional.

**Inputs:**
- `data/processed/vocabularies.pkl`, `data/processed/interactions_prior.npz`
- `data/processed/{train,val,test}_pairs.pkl`

**Outputs:**
- `models/item_based_cf/item_similarity.npz` — matriz esparsa de similaridade item-item (top-N catálogo)
- `models/item_based_cf/metrics.json`

**Etapas Principais:**
1. Carregar matriz esparsa `(n_users, n_items)` do histórico `prior`
2. Calcular similaridade item-item (cosine, via `sklearn.metrics.pairwise.cosine_similarity` sobre a matriz transposta esparsa)
3. **Random search de `top_m`** (vizinhos mais similares mantidos por item, truncando o resto) na validação — candidatos em `configs/model_config.yaml`
4. Para cada usuário, gerar score de recomendação somando a similaridade dos itens já comprados, ponderada pela frequência de compra
5. Avaliar nos splits de validação e teste com o melhor `top_m`
6. Persistir matriz de similaridade e métricas (incluindo os resultados do search)

**Dependências:**
- scikit-learn (`cosine_similarity`)
- scipy.sparse, numpy, pandas

**Rastreamento MLflow:**
- Log de params: `similarity_metric`, `k`, `top_m`, `dataset_hash`
- Log de métricas: `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `map_at_k` (val e teste, run final + runs de search)
- Log de artefatos: `item_similarity.npz`, `metrics.json`

---

## 5. Collaborative Filtering User-Based / KNN (`05_user_based_cf.ipynb`)

**Propósito:** Recomendar produtos comprados por usuários com histórico similar (k-vizinhos mais próximos), para avaliar se a personalização por similaridade de usuário compensa o custo computacional adicional frente ao item-based.

**Inputs:**
- `data/processed/vocabularies.pkl`, `data/processed/interactions_prior.npz`
- `data/processed/{train,val,test}_pairs.pkl`

**Outputs:**
- `models/user_based_cf/knn_model.pkl` — índice `NearestNeighbors` ajustado
- `models/user_based_cf/metrics.json`

**Etapas Principais:**
1. Carregar matriz esparsa `(n_users, n_items)` do histórico `prior`
2. Ajustar `NearestNeighbors(metric="cosine")` sobre o pool de usuários `prior`
3. **Random search de `n_neighbors`** na amostra de validação — candidatos em `configs/model_config.yaml`, reaproveitando o mesmo modelo ajustado (`n_neighbors` é passado por chamada)
4. Para cada usuário avaliado (val/teste), buscar os `n_neighbors` (melhor valor) vizinhos mais similares e agregar os itens comprados por eles, ponderados pela similaridade
5. **Nota de escopo:** a busca de vizinhos é executada sobre uma amostra fixa (seed 42) do pool de usuários `prior` e dos usuários avaliados, documentada em `metrics.json`, para manter o tempo de execução viável — comparável em espírito ao corte de catálogo da seção "Escopo de Catálogo".
6. Avaliar nos splits de validação e teste
7. Persistir modelo KNN e métricas (incluindo os resultados do search)

**Dependências:**
- scikit-learn (`NearestNeighbors`)
- scipy.sparse, numpy, pandas

**Rastreamento MLflow:**
- Log de params: `n_neighbors`, `neighbor_pool_size`, `eval_sample_size`, `dataset_hash`
- Log de métricas: `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `map_at_k` (val e teste, run final + runs de search)
- Log de artefatos: `knn_model.pkl`, `metrics.json`

---

## 6. Matrix Factorization (`06_matrix_factorization.ipynb`)

**Propósito:** Fatorar a matriz usuário-item em fatores latentes via `TruncatedSVD` (Scikit-Learn). Funciona como ponte conceitual entre o CF clássico (seções 4–5) e os embeddings aprendidos pelo NCF (seção 7), fortalecendo a análise de trade-off no Model Card.

**Inputs:**
- `data/processed/vocabularies.pkl`, `data/processed/interactions_prior.npz`
- `data/processed/{train,val,test}_pairs.pkl`

**Outputs:**
- `models/matrix_factorization/user_factors.npy`, `models/matrix_factorization/item_factors.npy`
- `models/matrix_factorization/metrics.json`

**Etapas Principais:**
1. Carregar matriz esparsa `(n_users, n_items)` do histórico `prior`
2. **Random search de `n_components`** na validação — candidatos em `configs/model_config.yaml`, um `TruncatedSVD` ajustado por candidato
3. Ajustar `TruncatedSVD(n_components=...)` final com o melhor candidato
4. Calcular score de afinidade usuário-produto via produto interno dos fatores latentes
5. Avaliar nos splits de validação e teste
6. Persistir fatores latentes e métricas (incluindo os resultados do search)

**Dependências:**
- scikit-learn (`TruncatedSVD`)
- scipy.sparse, numpy, pandas

**Rastreamento MLflow:**
- Log de params: `n_components`, `dataset_hash`
- Log de métricas: `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `map_at_k` (val e teste, run final + runs de search), `explained_variance_ratio`
- Log de artefatos: `user_factors.npy`, `item_factors.npy`, `metrics.json`

---

## 7. NCF Training - Treinamento do Modelo Principal (`07_ncf_training.ipynb`)

**Propósito:** Treinar o modelo neural principal — **Neural Collaborative Filtering (NCF)** — em PyTorch.

### 7.1 Justificativa da Arquitetura

O dataset Instacart é de **feedback implícito** (compra ou não-compra; sem nota explícita). Para esse cenário, NCF é a escolha mais adequada porque:

- Aprende embeddings de usuário e item diretamente do padrão de co-ocorrência em cestas, sem exigir features manuais.
- Generaliza melhor que CF clássico (KNN/popularidade) para pares usuário-produto nunca vistos, ao combinar a interação latente (produto interno) com uma camada MLP capaz de capturar interações não lineares.
- É a interpretação padrão de "embedding-based" exigida pelo edital, e mantém comparabilidade direta com os baselines de CF (mesma família conceitual, complexidade crescente).

### 7.2 Arquitetura

```
user_id ──> Embedding(n_users, embedding_dim) ──┐
                                                  ├─> concat ──> MLP (Linear+ReLU+Dropout)* ──> Linear(1) ──> sigmoid ──> score
item_id ──> Embedding(n_items, embedding_dim) ──┘
```

- `embedding_dim`: 32–64 (hiperparâmetro rastreado no MLflow).
- MLP: 2–3 camadas ocultas (ex.: 128 → 64 → 32), `ReLU` + `Dropout(0.2)`.
- Saída: score de afinidade usuário-produto em `(0, 1)`.

```python
import torch
import torch.nn as nn


class NeuralCollaborativeFiltering(nn.Module):
    """Modelo NCF para recomendação com feedback implícito."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Calcula o score de afinidade usuário-produto."""
        x = torch.cat([self.user_embedding(user_ids), self.item_embedding(item_ids)], dim=1)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)
```

### 7.3 Estratégia de Negative Sampling

O dataset só contém pares positivos (produtos efetivamente comprados). Sem exemplos negativos, o modelo não aprende a discriminar. Estratégia:

1. Para cada par positivo `(user, item)` no batch, amostrar `N` itens negativos (`N` = 4–10, hiperparâmetro `negative_ratio`) **uniformemente** sobre o catálogo.
2. Treinar com **Binary Cross-Entropy**: rótulo `1` para positivos, `0` para negativos amostrados.
3. Resampling a cada época (não fixar os mesmos negativos), para reduzir variância e overfitting em negativos específicos.
4. `RANDOM_SEED = 42` fixado também na amostragem de negativos, para reprodutibilidade.

**Nota de decisão (revisão pós-experimento):** a primeira versão usava negative sampling ponderado pela popularidade do item. Isso se mostrou contraproducente: como o Instacart tem forte viés de recompra de itens populares, penalizar repetidamente esses itens como negativos ensinou o modelo a *suprimir* exatamente os itens mais relevantes, fazendo o NCF performar pior que o baseline de popularidade. A amostragem foi trocada para uniforme.

### 7.4 Random Search de Hiperparâmetros (Validação)

Busca aleatória sobre `learning_rate`, `embedding_dim` e `hidden_dims` (candidatos em `configs/model_config.yaml`), com poucas épocas (`search_epochs`/`search_early_stopping_patience`) para manter o custo viável em CPU. Avaliada apenas na amostra de validação (nunca no teste). O treinamento final (mais épocas, `early_stopping_patience` completo) usa os melhores hiperparâmetros encontrados.

**Inputs:**
- `data/processed/vocabularies.pkl`, `data/processed/interactions_prior.npz`
- `data/processed/{train,val}_pairs.pkl`
- Configurações em `configs/model_config.yaml` (negative_ratio, batch_size, epochs, early_stopping_patience, e o espaço de busca em `ncf.search`)

**Outputs:**
- Modelo treinado: `models/neural_network/model.pt`
- Histórico de treinamento: `models/neural_network/training_history.pkl`
- Checkpoint melhor modelo: `models/neural_network/best_model.pt`

**Etapas Principais:**
1. Carregamento de vocabulários e pares de treino/validação processados
2. Random search de hiperparâmetros na amostra de validação (poucas épocas por trial)
3. Treinamento final com os melhores hiperparâmetros (mais épocas, early stopping completo)
4. Loop de treinamento com validação (Recall@K/NDCG@K) por época
5. Early stopping baseado em métrica de validação
6. Salvamento de checkpoints
7. Plotagem de curvas de treinamento/validação
8. Avaliação final completa em validação e teste (população total)

**Dependências:**
- torch, torch.nn
- pandas, numpy
- matplotlib

**Rastreamento MLflow:**
- Log de params (por trial de search e na run final): learning_rate, embedding_dim, hidden_dims, negative_ratio, batch_size, dataset_hash
- Log de métricas: train_loss, val_recall_at_k_sample, val_ndcg_at_k_sample (por época); val_precision_at_k, val_recall_at_k, val_ndcg_at_k, val_map_at_k e equivalentes de teste (avaliação final completa)
- Log de artefatos: model.pt, training_history.pkl
- Modelo registrado no Model Registry (stage: Staging)

---

## 8. Model Comparison - Avaliação e Comparação Final (`08_model_comparison.ipynb`)

**Propósito:** Avaliar todos os modelos (popularidade, item-CF, user-CF, matrix factorization, NCF) no mesmo split de teste e gerar o relatório/Model Card final.

### 8.1 Métricas Oficiais (≥ 4 exigidas pelo edital)

Todas calculadas por usuário no conjunto de teste e agregadas pela média (`K` configurável, padrão `K=10`).

| Métrica | Fórmula (por usuário) | O que mede |
|---------|------------------------|------------|
| **Precision@K** | `acertos_top_k / K` | Quão "limpo" é o top-K: proporção de recomendações relevantes. |
| **Recall@K** | `acertos_top_k / total_itens_relevantes` | Cobertura: quanto do que o usuário realmente comprou foi recuperado. |
| **NDCG@K** | `DCG@K / IDCG@K`, onde `DCG@K = Σ rel_i / log2(i+1)` | Qualidade do ranking, penalizando acertos em posições baixas. |
| **MAP@K** | média de `Precision@i` em cada posição `i` onde há acerto, até `K` | Agrega precisão considerando a ordem dos acertos. |

```python
def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calcula recall@k para uma lista de recomendações."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)
```

### 8.2 Métricas Complementares (Model Card)

| Métrica | O que mede | Por que registrar |
|---------|------------|---------------------|
| **Hit Rate@K** | % de usuários com ≥ 1 acerto no top-K | Mais intuitiva para apresentação (vídeo STAR). |
| **Coverage** | % do catálogo de produtos recomendado pelo menos uma vez | Detecta viés de recomendar sempre os mesmos itens populares. |
| **Tempo de inferência (ms/request)** | Latência média de geração de recomendações | Relevante para a seção opcional de deploy. |
| **Tempo de treinamento / uso de memória** | Custo computacional de cada modelo | Justifica o trade-off complexidade vs. ganho no Model Card. |
| **Tamanho do modelo serializado** | Impacto no tamanho da imagem Docker | Trade-off complexidade vs. portabilidade. |

### 8.3 Critério de Promoção no Model Registry

Os 5 modelos são registrados como MLflow Models (`popularity_recommender`, `item_based_cf_recommender`, `user_based_cf_recommender`, `matrix_factorization_recommender`, `ncf_recommender`) — os 4 baselines via wrappers `mlflow.pyfunc.PythonModel` (padrão Strategy, mesma interface `predict`), o NCF via `mlflow.pytorch.log_model` no notebook 07. O modelo com maior `Recall@K` no split de teste é promovido a `Production`; todos os demais permanecem em `Staging`. Essa regra é independente de qual modelo vence — se o NCF (modelo principal exigido em PyTorch) não superar os baselines, ele documenta o trade-off no Model Card mas não impede que o melhor modelo real seja promovido, conforme exigido pelo edital ("modelo promovido a Production").

**Inputs:**
- `data/processed/test_pairs.pkl`
- Artefatos de cada modelo: `models/baseline_popularity/`, `models/item_based_cf/`, `models/user_based_cf/`, `models/matrix_factorization/`, `models/neural_network/model.pt`

**Outputs:**
- Métricas finais: `models/evaluation/metrics_comparison.csv`
- Curvas de performance: `models/evaluation/plots/`
- Model Card: `models/MODEL_CARD.md`
- Relatório de erros: `models/evaluation/error_analysis.pkl`

**Etapas Principais:**
1. Carregamento de todos os modelos (popularidade, item-CF, user-CF, matrix factorization, NCF)
2. Geração de previsões no split de teste, para todos os modelos
3. Cálculo das métricas oficiais (Precision@K, Recall@K, NDCG@K, MAP@K) por modelo
4. Cálculo das métricas complementares (Hit Rate@K, Coverage, latência, tamanho do modelo)
5. Comparação visual e estatística entre os 5 modelos
6. Análise de erros do modelo principal (NCF)
7. Detecção de possíveis vieses (ex.: concentração em produtos populares)
8. Geração de Model Card com limitações
9. Decisão de promoção no Registry conforme critério da seção 8.3

**Dependências:**
- torch, scikit-learn
- pandas, numpy
- matplotlib, seaborn

**Rastreamento MLflow:**
- Log de métricas oficiais: precision_at_k, recall_at_k, ndcg_at_k, map_at_k (por modelo)
- Log de métricas complementares: hit_rate_at_k, coverage, inference_latency_ms, model_size_mb
- Log de artefatos: metrics_comparison.csv, MODEL_CARD.md, plots/
- Promoção de modelo no Registry (stage: Production) conforme critério definido em 8.3

---

## Fluxo de Execução Recomendado

```
01_eda.ipynb
    ↓
02_preprocessing.ipynb
    ↓
03_baseline_popularity.ipynb ─┐
04_item_based_cf.ipynb ───────┤
05_user_based_cf.ipynb ────────┤  (independentes entre si, todos consomem 02_preprocessing.ipynb)
06_matrix_factorization.ipynb ┤
07_ncf_training.ipynb ────────┘
    ↓
08_model_comparison.ipynb
```

**Notas:**
- Todos os notebooks devem ter RANDOM_SEED = 42 fixado
- print() é permitido em notebooks para uso exploratório; em código movido para src/ usar structlog
- Cada notebook deve ter uma célula de configuração inicial (imports, seed, logging)
- Outputs devem ser salvos de forma versionada (com timestamp ou git hash)

---

## Integração com MLflow e DVC

**MLflow:**
- Cada notebook executa múltiplos runs
- Rastreamento centralizado em `mlruns/` (gitignored)
- Modelos finais promocionados via Model Registry

**DVC:**
- Dados processados são versionados via DVC
- Pipeline reprodutível em `dvc.yaml` orquestra notebooks
- `dvc.lock` garante reproducibilidade

---

## Estrutura de Diretórios para Outputs

```
notebooks/
├── 01_eda.ipynb
├── 02_preprocessing.ipynb
├── 03_baseline_popularity.ipynb
├── 04_item_based_cf.ipynb
├── 05_user_based_cf.ipynb
├── 06_matrix_factorization.ipynb
├── 07_ncf_training.ipynb
├── 08_model_comparison.ipynb
└── outputs/
    ├── eda/
    │   └── plots/
    └── model_comparison/
        └── plots/

models/
├── baseline_popularity/
├── item_based_cf/
├── user_based_cf/
├── matrix_factorization/
├── neural_network/
└── evaluation/
```

---

## Checklist de Qualidade para Notebooks

- [ ] Seed fixado (RANDOM_SEED = 42)
- [ ] Imports organizados e comentados
- [ ] Logging estruturado via structlog
- [ ] print() restrito a uso exploratório (código movido para src/ usa structlog)
- [ ] Tipo hints em funções customizadas
- [ ] Docstrings em padrão Google Style
- [ ] Outputs salvos com versionagem
- [ ] MLflow rastreamento implementado
- [ ] Reprodutibilidade garantida
- [ ] Código refatorado em funções (máx 20 linhas)
