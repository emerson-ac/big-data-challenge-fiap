# Notebooks do Projeto

Este arquivo documenta a estrutura, propósito e fluxo de execução dos notebooks utilizados no projeto de sistema de recomendação.

---

## Visão Geral

Os notebooks são organizados por fase do projeto e devem ser executados em ordem específica para garantir reprodutibilidade e rastreabilidade das análises.

**Estrutura de Diretórios:**
```
notebooks/
├── 01_eda.ipynb              # Análise exploratória de dados
├── 02_baseline.ipynb         # Modelos baseline para comparação
├── 03_preprocessing.ipynb    # Pré-processamento e feature engineering
├── 04_model_training.ipynb   # Treinamento do modelo principal
└── 05_evaluation.ipynb       # Avaliação e comparação de modelos
```

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
- matplotlib, seaborn, plotly
- scipy (para testes estatísticos)

**Rastreamento MLflow:** Não aplicável (exploração somente)

---

## 2. Baseline - Modelos de Referência (`02_baseline.ipynb`)

**Propósito:** Estabelecer modelos simples como benchmark para comparação com o modelo principal.

**Inputs:**
- `data/raw/` (dados brutos)
- Saídas do notebook 01_eda.ipynb (insights sobre distribuições)

**Outputs:**
- Modelos baseline salvos em `models/baseline/`
- Métricas de performance em formato CSV: `models/baseline/metrics.csv`
- Previsões de teste: `data/processed/baseline_predictions.pkl`

**Baselines a Implementar:**
1. **Popularity-Based:** Recomenda itens mais populares globalmente
2. **Collaborative Filtering (User-Based):** Similitude entre usuários via cosine distance
3. **Collaborative Filtering (Item-Based):** Similitude entre itens via matriz de co-compra
4. **Nearest Neighbors (KNN):** k-NN com métrica customizada

**Etapas Principais:**
1. Preparação e split de dados (train/test estratificado)
2. Implementação de cada baseline
3. Avaliação com 4+ métricas (Precision@K, Recall@K, NDCG, MAP)
4. Comparação e documentação de resultados
5. Armazenamento de modelos com pickle/joblib

**Dependências:**
- scikit-learn
- pandas, numpy
- scipy

**Rastreamento MLflow:**
- Log de params: modelo, K, métrica
- Log de métricas: precision@k, recall@k, ndcg, map
- Log de artefato: predictions.pkl, model.pkl

---

## 3. Preprocessing - Preparação de Dados (`03_preprocessing.ipynb`)

**Propósito:** Transformar dados brutos em features apropriadas para o modelo neural.

**Inputs:**
- `data/raw/` (dados brutos)
- Saídas do notebook 01_eda.ipynb (conhecimento sobre dados)

**Outputs:**
- Dataset processado: `data/processed/train_processed.pkl`
- Dataset de teste: `data/processed/test_processed.pkl`
- Vocabulário de usuários/itens: `data/processed/vocabularies.pkl`
- Scaler/Normalizer: `data/processed/scaler.pkl`

**Etapas Principais:**
1. Tratamento de valores ausentes
2. Remoção de outliers (usuários/itens anômalos)
3. Codificação de usuários e itens (mapeamento numérico)
4. Engenharia de features (frequência de compra, recência, RFM)
5. Normalização/Padronização de features numéricas
6. Split estratificado de dados (train/val/test)
7. Salvamento de vocabulários e transformadores

**Dependências:**
- pandas, numpy, sklearn
- pickle

**Rastreamento MLflow:**
- Log de dataset hash (SHA256)
- Log de parâmetros: train/val/test split ratio
- Log de artefatos: vocabularies.pkl, scaler.pkl

---

## 4. Model Training - Treinamento do Modelo Principal (`04_model_training.ipynb`)

**Propósito:** Treinar a rede neural (MLP ou Embedding-Based) em PyTorch.

**Inputs:**
- `data/processed/train_processed.pkl`
- `data/processed/val_processed.pkl`
- Configurações em `configs/model_config.yaml`

**Outputs:**
- Modelo treinado: `models/neural_network/model.pt`
- Histórico de treinamento: `models/neural_network/training_history.pkl`
- Checkpoint melhor modelo: `models/neural_network/best_model.pt`

**Etapas Principais:**
1. Carregamento de dados processados
2. Construção da arquitetura neural (MLP ou Embedding)
3. Definição de loss function e optimizer
4. Loop de treinamento com validação
5. Early stopping baseado em métrica de validação
6. Salvamento de checkpoints
7. Plotagem de curvas de treinamento/validação

**Dependências:**
- torch, torch.nn
- pandas, numpy
- matplotlib

**Rastreamento MLflow:**
- Log de params: learning_rate, batch_size, epochs, embedding_dim, hidden_layers
- Log de métricas: train_loss, val_loss, val_ndcg (por época)
- Log de artefatos: model.pt, training_history.pkl
- Modelo registrado no Model Registry (stage: Staging)

---

## 5. Evaluation - Avaliação e Comparação (`05_evaluation.ipynb`)

**Propósito:** Avaliar o modelo principal contra baselines e gerar relatório final.

**Inputs:**
- `data/processed/test_processed.pkl`
- Modelos: `models/baseline/`, `models/neural_network/model.pt`
- Previsões: `data/processed/baseline_predictions.pkl`

**Outputs:**
- Métricas finais: `models/evaluation/metrics_comparison.csv`
- Curvas de performance: `models/evaluation/plots/`
- Model Card: `models/MODEL_CARD.md`
- Relatório de erros: `models/evaluation/error_analysis.pkl`

**Etapas Principais:**
1. Carregamento de todos os modelos (baselines + neural)
2. Geração de previsões em dados de teste
3. Cálculo de 4+ métricas por modelo
4. Comparação visual e estatística
5. Análise de erros do modelo principal
6. Detecção de possíveis vieses
7. Geração de Model Card com limitações
8. Recomendação de melhor modelo

**Dependências:**
- torch, sklearn
- pandas, numpy
- matplotlib, seaborn

**Rastreamento MLflow:**
- Log de métricas finais: precision@k, recall@k, ndcg, map (por modelo)
- Log de artefatos: metrics_comparison.csv, MODEL_CARD.md, plots/
- Promoção de modelo no Registry (stage: Production) se performance adequada

---

## Fluxo de Execução Recomendado

```
01_eda.ipynb
    ↓
02_baseline.ipynb (paralelo com 03)
    ↓
03_preprocessing.ipynb
    ↓
04_model_training.ipynb
    ↓
05_evaluation.ipynb
```

**Notas:**
- Todos os notebooks devem ter RANDOM_SEED = 42 fixado
- Nenhum notebook deve conter print(); usar logging via structlog
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
├── 02_baseline.ipynb
├── 03_preprocessing.ipynb
├── 04_model_training.ipynb
├── 05_evaluation.ipynb
└── outputs/
    ├── eda/
    │   ├── plots/
    │   └── data_profile.json
    ├── baseline/
    │   └── metrics.csv
    └── evaluation/
        ├── plots/
        └── metrics_comparison.csv
```

---

## Checklist de Qualidade para Notebooks

- [ ] Seed fixado (RANDOM_SEED = 42)
- [ ] Imports organizados e comentados
- [ ] Logging estruturado via structlog
- [ ] Sem uso de print()
- [ ] Tipo hints em funções customizadas
- [ ] Docstrings em padrão Google Style
- [ ] Outputs salvos com versionagem
- [ ] MLflow rastreamento implementado
- [ ] Reprodutibilidade garantida
- [ ] Código refatorado em funções (máx 20 linhas)
