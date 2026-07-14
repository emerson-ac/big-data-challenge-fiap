# Deploy da API (FastAPI) no cluster EKS

Deploy da **API REST de recomendação** (`src/api/main.py`, FastAPI/uvicorn) como
um Deployment Kubernetes nativo — **Deployment + Service + Ingress (ALB) + HPA**.

## O que é provisionado

| Recurso | Origem |
|---|---|
| Cluster EKS `recsys-challenge` | `cluster/cluster.yaml` (eksctl) |
| Namespace `models` + SA `sa-model-s3` (IRSA, S3 read-only) | `cluster/cluster.yaml` |
| AWS Load Balancer Controller (Ingress `alb`) | `scripts/deploy.sh` |
| metrics-server (para o HPA) | addon em `cluster/cluster.yaml` |

## Artefatos via S3 (não embutidos na imagem)

Um `initContainer` (`aws-cli`) baixa os artefatos do modelo do bucket para volumes
`emptyDir` montados em `/app/models` e `/app/data` — os caminhos que o
`RecommendationEngine` (`src/models/inference.py`) espera a partir de `/app`.
As credenciais vêm do IRSA da SA `sa-model-s3` (sem chaves no pod).

Layout no bucket (criado por `scripts/publish_api_artifacts.sh`):

```
s3://<bucket>/recsys-api/models/item_based_cf/item_similarity.npz
s3://<bucket>/recsys-api/models/baseline_popularity/ranking.pkl
s3://<bucket>/recsys-api/data/processed/vocabularies.pkl
s3://<bucket>/recsys-api/data/processed/interactions_prior.npz
```

## Como publicar

```bash
# 0. Pré-requisitos: cluster provisionado (bash scripts/bootstrap.sh)
#    + plataforma instalada (bash scripts/deploy.sh step 1).

# 1. Gere os artefatos (com dados reais em data/raw/)
uv run dvc repro

# 2. Envie os artefatos para o S3
bash scripts/publish_api_artifacts.sh

# 3. Build e push da imagem da API
docker build -t <REGISTRY>/recsys-api:latest .
docker push <REGISTRY>/recsys-api:latest

# 4. Ajuste a imagem em deployment.yaml (campo `image:`) e aplique
kubectl apply -k k8s/api

# 5. Acompanhe o rollout e descubra o hostname do ALB
kubectl -n models rollout status deploy/recsys-api
kubectl -n models get ingress recsys-api \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'; echo
```

## Testar

```bash
ALB_HOST="<hostname-do-alb>"
curl "http://${ALB_HOST}/health/status"
curl -X POST "http://${ALB_HOST}/recommendations/" \
  -H "Content-Type: application/json" -d '{"user_id":1,"top_k":5}'
```

## Ajustes por ambiente

| Onde | O quê |
|---|---|
| `deployment.yaml` | `image` (registry), `MODELS_BUCKET`, recursos |
| `ingress.yaml` | `certificate-arn` (ACM) para HTTPS, host opcional |
| `scripts/publish_api_artifacts.sh` | `MODELS_BUCKET` / `S3_PREFIX` (via env) |
