# Model Card - Sistema de Recomendacao Instacart

## Visao Geral

Comparacao de 5 modelos de recomendacao de produtos (top-10) avaliados no
split de teste interno derivado do dataset Instacart. Catalogo restrito aos
3000 produtos mais comprados (ver `docs/NOTEBOOKS.md`, secao "Escopo de
Catalogo"). Dataset hash: `c3fda3f4a8d64ffd...`.

## Metricas (split de teste)

| model                |   precision_at_k |   recall_at_k |   ndcg_at_k |   map_at_k |   hit_rate_at_k |   coverage_at_k |   inference_latency_ms |   model_size_mb |
|:---------------------|-----------------:|--------------:|------------:|-----------:|----------------:|----------------:|-----------------------:|----------------:|
| popularity           |           0.0747 |        0.0935 |      0.1097 |     0.0522 |          0.4729 |          0.0033 |                 0.0001 |          0.023  |
| item_based_cf        |           0.1292 |        0.2236 |      0.2172 |     0.1274 |          0.6722 |          0.8073 |                 0.0815 |         29.0345 |
| user_based_cf        |           0.127  |        0.2051 |      0.2117 |     0.1227 |          0.6657 |          0.7297 |                 0.5362 |          4.3148 |
| matrix_factorization |           0.1398 |        0.2017 |      0.2032 |     0.1137 |          0.6636 |          0.2153 |                 0.0344 |         74.0481 |
| ncf                  |           0.075  |        0.093  |      0.1093 |     0.0519 |          0.4726 |          0.0143 |                 4.9801 |         31.8805 |

## Decisao de Promocao

- NCF supera todos os baselines em recall@k: False
- NCF supera todos os baselines em ndcg@k: False
- Modelo com melhor recall@k nesta rodada: **item_based_cf**
- `item_based_cf_recommender` promovido a **Production** no
  MLflow Model Registry; os demais modelos, incluindo `ncf_recommender`,
  permanecem em **Staging**.

## Limitacoes e Vieses

1. **Catalogo restrito**: avaliacao limitada aos top-3000 produtos por
   volume de compras (cobertura de 73.1% do
   volume total). Produtos de nicho fora desse conjunto nunca sao
   recomendados nem avaliados.
2. **Amostragem no user-based CF**: por custo computacional da busca de
   vizinhos, o user-based CF foi avaliado em uma amostra de
   3,000 usuarios de teste, nao na populacao completa como os
   demais modelos — os valores nao sao estritamente comparaveis ponto a
   ponto, apenas direcionalmente.
3. **Negative sampling do NCF sem deduplicacao exaustiva**: a amostragem
   uniforme de negativos nao verifica exaustivamente se o item sorteado ja
   foi comprado pelo usuario, introduzindo ruido controlado no sinal de
   treino. Amostragem ponderada por popularidade foi testada e descartada
   (ver `docs/NOTEBOOKS.md`, secao 7.3): nesse dataset, com forte vies de
   recompra, ela penalizava justamente os itens mais relevantes.
4. **Vies de popularidade**: todos os modelos tendem a favorecer produtos de
   alta frequencia (coverage@k tipicamente baixo frente ao catalogo total),
   refletindo o proprio comportamento de recompra do dataset Instacart.
5. **Orcamento de treino do NCF**: treinado em CPU com early stopping
   (paciencia configurada em `configs/model_config.yaml`); mais epocas ou
   ajuste fino de hiperparametros podem alterar a posicao relativa do NCF
   frente aos baselines classicos.

## Recomendacao

Modelo recomendado para producao nesta rodada: **item_based_cf**, promovido a
`Production` no MLflow Model Registry (`item_based_cf_recommender`).
O `ncf_recommender` (modelo principal exigido em PyTorch) permanece em
`Staging` e pode ser promovido em rodadas futuras caso supere o desempenho
atual dos baselines classicos.
