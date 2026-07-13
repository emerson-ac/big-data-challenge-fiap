# Model Card - Sistema de Recomendacao Instacart

Comparacao de 5 modelos (top-10) no split de teste interno. Dataset hash: `c3fda3f4a8d64ffd...`.

## Metricas (split de teste)

| model                |   precision_at_k |   recall_at_k |   ndcg_at_k |   map_at_k |   hit_rate_at_k |   coverage_at_k |   inference_latency_ms |   model_size_mb |
|:---------------------|-----------------:|--------------:|------------:|-----------:|----------------:|----------------:|-----------------------:|----------------:|
| popularity           |           0.0747 |        0.0935 |      0.1097 |     0.0522 |          0.4729 |          0.0033 |                 0.0003 |          0.023  |
| item_based_cf        |           0.1292 |        0.2236 |      0.2172 |     0.1274 |          0.6722 |          0.8073 |                 0.1739 |         29.0345 |
| user_based_cf        |           0.134  |        0.2222 |      0.2224 |     0.1304 |          0.689  |          0.7307 |                 0.3282 |          4.3496 |
| matrix_factorization |           0.1398 |        0.2017 |      0.2032 |     0.1137 |          0.6636 |          0.2153 |                 0.0184 |         74.0481 |
| ncf                  |           0.0751 |        0.0932 |      0.1098 |     0.0523 |          0.4723 |          0.01   |                 4.5397 |         63.7616 |

## Decisao de Promocao

- NCF supera todos os baselines em recall@k: False
- NCF supera todos os baselines em ndcg@k: False
- Modelo com melhor recall@k: **item_based_cf**
- `item_based_cf_recommender` promovido via alias **@production** no MLflow Model
  Registry (MLflow 3 usa aliases no lugar de stages); os demais não são registrados.

Gerado automaticamente por `src/pipeline/evaluate.py`.
