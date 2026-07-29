#!/usr/bin/env bash
# Passo one-shot (imperativo): cria o cluster EKS e o bucket S3 dos modelos.
# Depois disto, tudo é declarativo via scripts/deploy.sh.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${EKS_CLUSTER:-recsys-challenge}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
BUCKET="${MODELS_BUCKET:-recsys-challenge-models-${ACCOUNT_ID}}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ">> Conferindo identidade AWS ativa..."
ACTIVE="$(aws sts get-caller-identity --query Account --output text)"
echo "   Conta: $ACTIVE | Cluster: $CLUSTER | Região: $REGION"

echo ">> Criando cluster EKS (leva ~15-20 min)..."
eksctl create cluster -f "$ROOT/cluster/cluster.yaml"

echo ">> Criando bucket S3 para modelos: $BUCKET"
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "   bucket já existe, seguindo."
else
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
fi

echo ">> Cluster e bucket prontos. Próximo passo: bash scripts/deploy.sh"
