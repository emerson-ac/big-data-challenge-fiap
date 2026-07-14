#!/usr/bin/env bash
# Aplica (declarativamente) tudo que roda DENTRO do cluster:
# AWS LB Controller -> API FastAPI (k8s/api/).
# Idempotente: pode rodar quantas vezes quiser. Usado localmente e pelo GitHub Actions.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${EKS_CLUSTER:-recsys-challenge}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LBC_CHART_VERSION="1.11.0"

echo ">> Atualizando kubeconfig..."
aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION"

echo ">> [1/2] AWS Load Balancer Controller $LBC_CHART_VERSION"
helm repo add eks https://aws.github.io/eks-charts >/dev/null 2>&1 || true
helm repo update >/dev/null
helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system --version "$LBC_CHART_VERSION" \
  -f "$ROOT/platform/aws-lb-controller/values.yaml"
kubectl -n kube-system rollout status deploy/aws-load-balancer-controller --timeout=180s

echo ">> [2/2] Aplicando API FastAPI (k8s/api/)"
kubectl apply -k "$ROOT/k8s/api"
kubectl -n models rollout status deploy/recsys-api --timeout=300s

echo ">> Estado final:"
kubectl -n models get deploy,svc,ingress recsys-api
echo "URL pública (ALB) — aguarde alguns minutos para provisionar:"
kubectl -n models get ingress recsys-api \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'; echo
