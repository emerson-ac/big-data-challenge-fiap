# Model Card - Sistema de Recomendacao Instacart

Comparacao de 5 modelos (top-10) no split de teste interno. Dataset hash: `ad3c4f16d15ed60d...`.

## Metricas (split de teste)

| model                |   precision_at_k |   recall_at_k |   ndcg_at_k |   map_at_k |   hit_rate_at_k |   coverage_at_k |   inference_latency_ms |   model_size_mb |
|:---------------------|-----------------:|--------------:|------------:|-----------:|----------------:|----------------:|-----------------------:|----------------:|
| popularity           |           0.2154 |        0.4589 |      0.4007 |     0.2438 |          0.9231 |           0.05  |                 0.0024 |          0.0017 |
| item_based_cf        |           0.2179 |        0.4625 |      0.3782 |     0.2277 |          0.9231 |           0.06  |                 0.0218 |          0.0322 |
| user_based_cf        |           0.1615 |        0.3336 |      0.2541 |     0.133  |          0.8205 |           0.28  |                 0.0314 |          0.0453 |
| matrix_factorization |           0.1487 |        0.295  |      0.2393 |     0.1252 |          0.7692 |           0.23  |                 0.0116 |          0.035  |
| ncf                  |           0.1744 |        0.3552 |      0.3493 |     0.2159 |          0.8462 |           0.095 |                 0.1592 |          0.3978 |

## Decisao de Promocao

- NCF supera todos os baselines em recall@k: False
- NCF supera todos os baselines em ndcg@k: False
- Modelo com melhor recall@k: **item_based_cf**
- `item_based_cf_recommender` promovido via stages Staging -> Production e
  alias **@production** no MLflow Model Registry; os demais nao sao registrados.

Gerado automaticamente por `src/pipeline/evaluate.py`.
