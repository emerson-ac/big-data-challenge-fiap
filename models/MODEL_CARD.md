# Model Card - Sistema de Recomendacao Instacart

Comparacao de 5 modelos (top-10) no split de teste interno. Dataset hash: `346a31d5508a2644...`.

## Metricas (split de teste)

| model                |   precision_at_k |   recall_at_k |   ndcg_at_k |   map_at_k |   hit_rate_at_k |   coverage_at_k |   inference_latency_ms |   model_size_mb |
|:---------------------|-----------------:|--------------:|------------:|-----------:|----------------:|----------------:|-----------------------:|----------------:|
| popularity           |           0.1917 |        0.3865 |      0.3653 |     0.2175 |          0.9167 |          0.0333 |                 0.0024 |          0.0024 |
| item_based_cf        |           0.1917 |        0.3865 |      0.3361 |     0.1939 |          0.9167 |          0.04   |                 0.0394 |          0.0476 |
| user_based_cf        |           0.1583 |        0.3011 |      0.2356 |     0.1225 |          0.75   |          0.1833 |                 0.0299 |          0.0664 |
| matrix_factorization |           0.1458 |        0.2771 |      0.2196 |     0.1106 |          0.7917 |          0.1433 |                 0.0102 |          0.0491 |
| ncf                  |           0.1708 |        0.3375 |      0.3391 |     0.2011 |          0.8333 |          0.05   |                 0.1976 |          0.443  |

## Decisao de Promocao

- NCF supera todos os baselines em recall@k: False
- NCF supera todos os baselines em ndcg@k: False
- Modelo com melhor recall@k: **popularity**
- Promocao ao MLflow Model Registry ocorre na etapa de serving (Etapa 4).

Gerado automaticamente por `src/pipeline/evaluate.py`.
