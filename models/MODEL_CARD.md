# Model Card - Sistema de Recomendacao Instacart

Comparacao de 5 modelos (top-10) no split de teste interno. Dataset hash: `ad3c4f16d15ed60d...`.

## Metricas (split de teste)

| model                |   precision_at_k |   recall_at_k |   ndcg_at_k |   map_at_k |   hit_rate_at_k |   coverage_at_k |   inference_latency_ms |   model_size_mb |
|:---------------------|-----------------:|--------------:|------------:|-----------:|----------------:|----------------:|-----------------------:|----------------:|
| popularity           |           0.2111 |        0.4968 |      0.4464 |     0.291  |          0.9722 |           0.05  |                 0.0027 |          0.0017 |
| item_based_cf        |           0.2111 |        0.4968 |      0.3856 |     0.2365 |          0.9722 |           0.075 |                 0.0231 |          0.0322 |
| user_based_cf        |           0.1556 |        0.3646 |      0.2758 |     0.149  |          0.8611 |           0.26  |                 0.0332 |          0.0453 |
| matrix_factorization |           0.1333 |        0.3099 |      0.2541 |     0.1454 |          0.8056 |           0.24  |                 0.011  |          0.035  |
| ncf                  |           0.2056 |        0.4699 |      0.4312 |     0.2832 |          0.9722 |           0.07  |                 0.1549 |          0.3978 |

## Decisao de Promocao

- NCF supera todos os baselines em recall@k: False
- NCF supera todos os baselines em ndcg@k: False
- Modelo com melhor recall@k: **popularity**
- `item_based_cf_recommender` (pyfunc servindo **popularity**) promovido via stages
  Staging -> Production e alias **@production** no MLflow Model Registry.

Gerado automaticamente por `src/pipeline/evaluate.py`.
