# Recsys — predictor KServe customizado (BÔNUS)

Serve o **RecommendationEngine** (modelo Production, item-based CF) via KServe,
reusando a mesma infra de cluster do modelo `sklearn-iris`. Como o recomendador
não é um `modelFormat` nativo do KServe, sobe como **custom container**.

## Como publicar

```bash
# 1. Gere os artefatos (data/processed + models/*) — com dados reais ou sintéticos
uv run dvc repro

# 2. Build da imagem (a partir da RAIZ do repositório)
docker build -f models/recsys/Dockerfile -t <REGISTRY>/recsys:latest .
docker push <REGISTRY>/recsys:latest

# 3. Ajuste a imagem no inferenceservice.yaml e o host/ACM no ingress.yaml, e aplique
kubectl apply -f models/recsys/inferenceservice.yaml
kubectl apply -f models/recsys/ingress.yaml
```

## Testar

```bash
curl -s https://recsys.<seu-dominio>/v1/models/recsys:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"user_id": 1, "k": 5}]}'
```

> **Nota:** os artefatos (`item_similarity.npz`, `vocabularies.pkl`, etc.) precisam
> estar na imagem ou montados via volume/`storageUri`. Neste PoC eles são copiados
> no build (após `dvc repro`). Em produção, prefira montar de um bucket/DVC remote.
