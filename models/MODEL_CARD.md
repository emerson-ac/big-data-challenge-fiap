# Model Card - Sistema de Recomendacao Instacart

Comparacao de 5 modelos (top-10) no split de teste interno. Dataset hash: `346a31d5508a2644...`.

## Metricas (split de teste)

| model                |   precision_at_k |   recall_at_k |   ndcg_at_k |   map_at_k |   hit_rate_at_k |   coverage_at_k |   inference_latency_ms |   model_size_mb |
|:---------------------|-----------------:|--------------:|------------:|-----------:|----------------:|----------------:|-----------------------:|----------------:|
| popularity           |           0.175  |        0.3626 |      0.3759 |     0.2453 |          0.8654 |          0.0333 |                 0.0016 |          0.0024 |
| item_based_cf        |           0.175  |        0.3626 |      0.3496 |     0.2176 |          0.8654 |          0.04   |                 0.0244 |          0.0476 |
| user_based_cf        |           0.1538 |        0.3217 |      0.2348 |     0.1176 |          0.8077 |          0.21   |                 0.0298 |          0.0664 |
| matrix_factorization |           0.1635 |        0.3409 |      0.269  |     0.146  |          0.8269 |          0.1633 |                 0.0096 |          0.0491 |
| ncf                  |           0.1712 |        0.3524 |      0.3564 |     0.2299 |          0.8462 |          0.05   |                 0.1964 |          0.443  |

## Decisao de Promocao

- NCF supera todos os baselines em recall@k: False
- NCF supera todos os baselines em ndcg@k: False
- Modelo com melhor recall@k: **popularity**
- Promocao ao MLflow Model Registry ocorre na etapa de serving (Etapa 4).

Gerado automaticamente por `src/pipeline/evaluate.py`.
