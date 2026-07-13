#!/usr/bin/env bash
# Passo one-shot (imperativo): cria o cluster EKS e o bucket S3 dos modelos.
# Depois disto, tudo é declarativo via scripts/deploy.sh (ou o pipeline).
set -euo pipefail

# Parametrizável por ambiente (.env / variáveis do CI); defaults = PoC original.
REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${EKS_CLUSTER:-arcobridgegitops}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-177300752486}"
BUCKET="${MODELS_BUCKET:-arcobridgegitops-models-${ACCOUNT_ID}}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ">> Conferindo identidade AWS ativa..."
ACTIVE="$(aws sts get-caller-identity --query Account --output text)"
if [[ "$ACTIVE" != "$ACCOUNT_ID" ]]; then
  echo "ERRO: conta AWS ativa ($ACTIVE) != esperada ($ACCOUNT_ID). Abortando." >&2
  exit 1
fi

echo ">> Criando cluster EKS (leva ~15-20 min)..."
eksctl create cluster -f "$ROOT/cluster/cluster.yaml"

echo ">> Criando bucket S3 para modelos: $BUCKET"
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "   bucket já existe, seguindo."
else
  # us-east-1 não usa LocationConstraint
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
fi

echo ">> Cluster e bucket prontos. Próximo passo: ./scripts/deploy.sh"
