#!/usr/bin/env bash
# Aplica (declarativamente) tudo que roda DENTRO do cluster:
# cert-manager -> AWS LB Controller -> KServe -> modelo (S3) -> InferenceService + Ingress.
# Idempotente: pode rodar quantas vezes quiser. Usado localmente e pelo GitHub Actions.
set -euo pipefail

# Parametrizável por ambiente (.env / variáveis do CI); defaults = PoC original.
REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${EKS_CLUSTER:-arcobridgegitops}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-177300752486}"
BUCKET="${MODELS_BUCKET:-arcobridgegitops-models-${ACCOUNT_ID}}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- versões fixadas (ajuste conforme necessário) ----
CERT_MANAGER_VERSION="v1.16.2"
LBC_CHART_VERSION="1.11.0"
KSERVE_VERSION="v0.14.1"

echo ">> Atualizando kubeconfig..."
aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION"

echo ">> [1/5] cert-manager $CERT_MANAGER_VERSION"
kubectl apply -f "https://github.com/cert-manager/cert-manager/releases/download/${CERT_MANAGER_VERSION}/cert-manager.yaml"
kubectl -n cert-manager rollout status deploy/cert-manager-webhook --timeout=180s

echo ">> [2/5] AWS Load Balancer Controller $LBC_CHART_VERSION"
helm repo add eks https://aws.github.io/eks-charts >/dev/null 2>&1 || true
helm repo update >/dev/null
helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system --version "$LBC_CHART_VERSION" \
  -f "$ROOT/platform/aws-lb-controller/values.yaml"
kubectl -n kube-system rollout status deploy/aws-load-balancer-controller --timeout=180s

echo ">> [3/5] KServe $KSERVE_VERSION (RawDeployment)"
# O chart do KServe instala o controller no namespace do release (default).
helm upgrade --install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version "$KSERVE_VERSION"
helm upgrade --install kserve oci://ghcr.io/kserve/charts/kserve --version "$KSERVE_VERSION" \
  --set kserve.controller.deploymentMode=RawDeployment
# Espera o controller ficar pronto — o webhook precisa ter endpoints antes de criar o InferenceService.
kubectl -n default rollout status deploy/kserve-controller-manager --timeout=300s
kubectl -n default wait --for=jsonpath='{.subsets[0].addresses[0].ip}' \
  endpoints/kserve-webhook-server-service --timeout=120s

# Usamos nosso próprio Ingress ALB; desabilita o ingress interno do KServe (classe istio).
kubectl apply -f "$ROOT/platform/kserve/ingress-config-patch.yaml"
kubectl -n default rollout restart deploy/kserve-controller-manager
kubectl -n default rollout status deploy/kserve-controller-manager --timeout=180s

echo ">> [4/5] Treinando e enviando o modelo iris para s3://$BUCKET/sklearn-iris/"
( cd "$ROOT/models/sklearn-iris" \
    && python train.py \
    && aws s3 cp model.joblib "s3://$BUCKET/sklearn-iris/model.joblib" )

echo ">> Anotando a SA do modelo para acesso S3 do KServe"
kubectl annotate sa sa-model-s3 -n models \
  serving.kserve.io/s3-region="$REGION" \
  serving.kserve.io/s3-usehttps="1" --overwrite

echo ">> [5/5] Aplicando InferenceService + Ingress"
kubectl apply -f "$ROOT/models/sklearn-iris/inferenceservice.yaml"
kubectl apply -f "$ROOT/models/sklearn-iris/ingress.yaml"
kubectl -n models wait --for=condition=Ready inferenceservice/sklearn-iris --timeout=300s || true

echo ">> Estado final:"
kubectl -n models get inferenceservice sklearn-iris
echo "URL pública (ALB) — aguarde alguns minutos para provisionar:"
kubectl -n models get ingress sklearn-iris -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'; echo
