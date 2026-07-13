# Deploy da API (FastAPI) no cluster EKS

Deploy da **API REST de recomendação** (`src/api/main.py`, FastAPI/uvicorn) como
um Deployment Kubernetes nativo — **Deployment + Service + Ingress (ALB) + HPA** —
reaproveitando o cluster `arcobridgegitops` e a plataforma do bônus KServe/EKS.

> Este é um caminho **separado** do predictor KServe em `models/recsys/`:
> ali serve-se o *modelo* no protocolo V1 do KServe; aqui serve-se a **API REST**
> (`/health`, `/recommendations/{user_id}`). Ambos rodam no mesmo cluster.

## O que é reutilizado do cluster existente

| Recurso | Origem |
|---|---|
| Cluster EKS `arcobridgegitops` | `cluster/cluster.yaml` (eksctl) |
| Namespace `models` + SA `sa-model-s3` (IRSA, S3 read-only) | `cluster/cluster.yaml` |
| AWS Load Balancer Controller (Ingress `alb`) | `scripts/deploy.sh` |
| metrics-server (para o HPA) | addon em `cluster/cluster.yaml` |
| Cert ACM `*.pocsarcotech.com` | mesmo do `ingress.yaml` do bônus |

## Artefatos via S3 (não embutidos na imagem)

Um `initContainer` (`aws-cli`) baixa os artefatos do modelo do bucket para volumes
`emptyDir` montados em `/app/models` e `/app/data` — os caminhos relativos que o
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
# 0. Pré-requisitos: cluster provisionado + scripts/deploy.sh já rodado
#    (cert-manager, LB Controller, namespace models/SA sa-model-s3).

# 1. Gere os artefatos (com dados reais em data/raw/)
uv run dvc repro

# 2. Envie os artefatos para o S3 (layout esperado pelo initContainer)
bash scripts/publish_api_artifacts.sh

# 3. Build e push da imagem da API (Dockerfile na RAIZ do repo → uvicorn :8000)
docker build -t <REGISTRY>/recsys-api:latest .
docker push <REGISTRY>/recsys-api:latest

# 4. Ajuste a imagem em deployment.yaml e o host em ingress.yaml, e aplique tudo
kubectl apply -k k8s/api

# 5. Acompanhe o rollout e descubra o hostname do ALB
kubectl -n models rollout status deploy/recsys-api
kubectl -n models get ingress recsys-api \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'; echo
```

Aponte o DNS `recsys-api.pocsarcotech.com` (registro CNAME/ALIAS) para o hostname
do ALB retornado acima.

## Testar

```bash
curl -s https://recsys-api.pocsarcotech.com/health
curl -s "https://recsys-api.pocsarcotech.com/recommendations/1?k=5"
```

## Ajustes por ambiente

| Onde | O quê |
|---|---|
| `deployment.yaml` | `image` (registry), `MODELS_BUCKET`, `S3_PREFIX`, recursos |
| `ingress.yaml` | `host`, `certificate-arn` (ACM) |
| `scripts/publish_api_artifacts.sh` | `MODELS_BUCKET` / `S3_PREFIX` (via env) |
