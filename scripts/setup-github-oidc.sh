#!/usr/bin/env bash
# Cria o provedor OIDC do GitHub Actions + uma IAM Role que o workflow assume.
# Sem chaves estáticas na AWS. Rode uma vez, localmente.
#
# Uso:
#   GITHUB_REPO="org/repositorio" ./scripts/setup-github-oidc.sh
set -euo pipefail

# Parametrizável por ambiente (.env / variáveis do CI); defaults = PoC original.
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-177300752486}"
ROLE_NAME="${OIDC_ROLE_NAME:-github-actions-${EKS_CLUSTER:-arcobridgegitops}}"
: "${GITHUB_REPO:?Defina GITHUB_REPO=org/repositorio}"

OIDC_HOST="token.actions.githubusercontent.com"
OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_HOST}"

echo ">> Garantindo provedor OIDC do GitHub..."
if ! aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1; then
  # A AWS valida o certificado do GitHub pela CA; o thumbprint abaixo é placeholder.
  aws iam create-open-id-connect-provider \
    --url "https://${OIDC_HOST}" \
    --client-id-list "sts.amazonaws.com" \
    --thumbprint-list "ffffffffffffffffffffffffffffffffffffffff"
fi

TRUST="$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "${OIDC_ARN}" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": { "${OIDC_HOST}:aud": "sts.amazonaws.com" },
      "StringLike":   { "${OIDC_HOST}:sub": "repo:${GITHUB_REPO}:*" }
    }
  }]
}
EOF
)"

echo ">> Criando/atualizando role $ROLE_NAME (trust -> $GITHUB_REPO)..."
if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name "$ROLE_NAME" --policy-document "$TRUST"
else
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST"
fi

# PoC: AdministratorAccess para permitir eksctl/kubectl/helm/S3. Restrinja depois.
aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess

echo
echo ">> Pronto. Configure no GitHub (Settings > Secrets and variables > Actions > Variables):"
echo "   AWS_ROLE_ARN = arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
