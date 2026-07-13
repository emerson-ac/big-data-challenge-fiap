#!/usr/bin/env bash
# Derruba o cluster para não gastar. A VPC pré-existente NÃO é removida
# (o eksctl não a criou, então não a deleta).
set -euo pipefail

# Parametrizável por ambiente (.env / variáveis do CI); defaults = PoC original.
REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${EKS_CLUSTER:-arcobridgegitops}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ">> Removendo Ingress primeiro (para o LB Controller deletar o ALB)..."
aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION" 2>/dev/null || true
kubectl delete -f "$ROOT/models/sklearn-iris/ingress.yaml" --ignore-not-found || true
sleep 20  # dá tempo do ALB ser removido antes de apagar a rede

echo ">> Deletando cluster EKS..."
eksctl delete cluster --name "$CLUSTER" --region "$REGION" --wait

echo ">> Feito. (Bucket S3 e VPC permanecem — remova manualmente se quiser.)"
