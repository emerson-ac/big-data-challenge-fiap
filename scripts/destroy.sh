#!/usr/bin/env bash
# Derruba o cluster para não gastar. A VPC criada pelo eksctl é removida junto.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${EKS_CLUSTER:-recsys-challenge}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ">> Removendo Ingress primeiro (para o LB Controller deletar o ALB)..."
aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION" 2>/dev/null || true
kubectl delete -k "$ROOT/k8s/api" --ignore-not-found || true
sleep 20

echo ">> Deletando cluster EKS..."
eksctl delete cluster --name "$CLUSTER" --region "$REGION" --wait

echo ">> Feito. (Bucket S3 permanece — remova manualmente se quiser.)"
