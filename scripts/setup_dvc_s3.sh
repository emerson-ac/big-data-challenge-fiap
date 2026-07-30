#!/usr/bin/env bash
# Configura o remote DVC em S3 e publica os dados/artefatos do cache.
#
# Pré-requisitos:
#   1. Credenciais AWS com acesso de escrita ao bucket (aws configure / SSO)
#   2. Artefatos no cache DVC local (uv run dvc repro já executado)
#      — ou dados em data/raw/ para gerar do zero
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
BUCKET="${MODELS_BUCKET:-arcobridgegitops-models-${ACCOUNT_ID}}"
DVC_URL="s3://${BUCKET}/dvc"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ">> Configurando remote DVC -> $DVC_URL"
uv run dvc remote add -d --force s3remote "$DVC_URL"
uv run dvc remote modify s3remote region "$REGION"

if [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
  echo "ERRO: data/raw/ está vazio. Baixe o dataset do Kaggle antes." >&2
  exit 1
fi

# Rastreia dados brutos se ainda não estiver
if [ ! -f data/raw.dvc ]; then
  echo ">> Rastreando dados brutos (dvc add data/raw)"
  uv run dvc add data/raw
fi

echo ">> Enviando ao S3 ($DVC_URL)"
uv run dvc push

echo ">> Concluído. Dados e artefatos no S3."
echo "   Para regenerar do zero: uv run dvc repro && uv run dvc push"
