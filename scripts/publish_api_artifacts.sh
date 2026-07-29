#!/usr/bin/env bash
# Publica os artefatos que a API precisa para o bucket S3, no layout esperado
# pelo initContainer do Deployment (k8s/api/deployment.yaml).
#
# Gere os artefatos antes (uv run dvc repro) — com dados reais em data/raw/.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
BUCKET="${MODELS_BUCKET:-recsys-challenge-models-${ACCOUNT_ID}}"
PREFIX="${S3_PREFIX:-recsys-api}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

required=(
  "models/item_based_cf/item_similarity.npz"
  "models/baseline_popularity/ranking.pkl"
  "data/processed/vocabularies.pkl"
  "data/processed/interactions_prior.npz"
)
for f in "${required[@]}"; do
  if [[ ! -f "$ROOT/$f" ]]; then
    echo "ERRO: artefato ausente: $f — rode 'uv run dvc repro' primeiro." >&2
    exit 1
  fi
done

echo ">> Enviando artefatos para s3://$BUCKET/$PREFIX/ (region $REGION)"
aws s3 cp "$ROOT/models/item_based_cf/item_similarity.npz" \
  "s3://$BUCKET/$PREFIX/models/item_based_cf/item_similarity.npz" --region "$REGION"
aws s3 cp "$ROOT/models/baseline_popularity/ranking.pkl" \
  "s3://$BUCKET/$PREFIX/models/baseline_popularity/ranking.pkl" --region "$REGION"
aws s3 cp "$ROOT/data/processed/vocabularies.pkl" \
  "s3://$BUCKET/$PREFIX/data/processed/vocabularies.pkl" --region "$REGION"
aws s3 cp "$ROOT/data/processed/interactions_prior.npz" \
  "s3://$BUCKET/$PREFIX/data/processed/interactions_prior.npz" --region "$REGION"

echo ">> Concluído. Artefatos em s3://$BUCKET/$PREFIX/"
