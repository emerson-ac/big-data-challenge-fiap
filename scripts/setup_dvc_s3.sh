#!/usr/bin/env bash
# Configura o remote DVC em S3 e publica os dados/artefatos reais.
#
# Pré-requisitos (uma vez, na sua máquina):
#   1. Baixar os CSVs do Instacart do Kaggle e extrair para data/raw/
#      (aisles, departments, order_products__prior, order_products__train,
#       orders, products)
#   2. Credenciais AWS com acesso de escrita ao bucket (aws configure / SSO).
#
# Depois disso, o CI (model-release.yml) faz `dvc pull` do S3 e treina no dado real.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-177300752486}"
BUCKET="${MODELS_BUCKET:-arcobridgegitops-models-${ACCOUNT_ID}}"
DVC_URL="s3://${BUCKET}/dvc"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ">> Garantindo o remote 's3remote' -> $DVC_URL"
uv run dvc remote add -d --force s3remote "$DVC_URL"
uv run dvc remote modify s3remote region "$REGION"

if [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
  echo "ERRO: data/raw/ está vazio. Baixe o dataset do Kaggle antes de continuar." >&2
  exit 1
fi

echo ">> Rastreando os dados brutos com DVC (para o CI conseguir 'dvc pull')"
# data/raw é dependência do estágio preprocess; rastreá-lo com dvc add o versiona
# no cache/remote (os outs do pipeline são cache:false e ficam no git).
uv run dvc add data/raw

echo ">> Reproduzindo o pipeline com os dados reais (preprocess -> train -> evaluate)"
uv run dvc repro

echo ">> Publicando os dados no S3 ($DVC_URL)"
uv run dvc push

echo ">> Pronto. Commite: data/raw.dvc, data/.gitignore e o dvc.lock atualizado."
echo "   O CI (model-release.yml) agora consegue 'dvc pull' os dados reais."
