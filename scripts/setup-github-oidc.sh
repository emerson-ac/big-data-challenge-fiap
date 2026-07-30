#!/usr/bin/env bash
# Cria o provedor OIDC do GitHub Actions + a IAM Role que o deploy-api assume.
# Sem chaves estáticas na AWS. Idempotente — já foi aplicado nesta conta.
#
# A role recebe o MÍNIMO necessário para o deploy da API e o DVC:
#   - IAM: eks:DescribeCluster apenas no cluster alvo (para o update-kubeconfig);
#   - IAM: leitura/escrita S3 restrita aos prefixos dvc/ e recsys-api/ do
#     bucket de modelos (o segundo e o layout que o initContainer da API le);
#   - EKS: access entry com AmazonEKSEditPolicy ESCOPADA ao namespace `models`.
# O cluster é compartilhado: a role não deve poder tocar em kube-system nem em
# outros namespaces.
#
# Uso:
#   GITHUB_REPO="emerson-ac/big-data-challenge-fiap" ./scripts/setup-github-oidc.sh
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
CLUSTER="${EKS_CLUSTER:-arcobridgegitops}"
NAMESPACE="${EKS_NAMESPACE:-models}"
ROLE_NAME="${OIDC_ROLE_NAME:-github-actions-recsys-api}"
BUCKET="${MODELS_BUCKET:-arcobridgegitops-models-${ACCOUNT_ID}}"
API_PREFIX="${S3_PREFIX:-recsys-api}"
: "${GITHUB_REPO:?Defina GITHUB_REPO=org/repositorio}"

OIDC_HOST="token.actions.githubusercontent.com"
OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_HOST}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

echo ">> Garantindo provedor OIDC do GitHub..."
if ! aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1; then
  aws iam create-open-id-connect-provider \
    --url "https://${OIDC_HOST}" \
    --client-id-list "sts.amazonaws.com" \
    --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1"
fi

# Trust restrito: só a branch main e o environment `production` deste repo.
TRUST="$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "${OIDC_ARN}" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": { "${OIDC_HOST}:aud": "sts.amazonaws.com" },
      "StringLike": {
        "${OIDC_HOST}:sub": [
          "repo:${GITHUB_REPO}:ref:refs/heads/main",
          "repo:${GITHUB_REPO}:environment:production"
        ]
      }
    }
  }]
}
EOF
)"

POLICY="$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "DescribeClusterForKubeconfig",
    "Effect": "Allow",
    "Action": "eks:DescribeCluster",
    "Resource": "arn:aws:eks:${REGION}:${ACCOUNT_ID}:cluster/${CLUSTER}"
  }]
}
EOF
)"

echo ">> Criando/atualizando role $ROLE_NAME (trust -> $GITHUB_REPO)..."
if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name "$ROLE_NAME" --policy-document "$TRUST"
else
  aws iam create-role --role-name "$ROLE_NAME" \
    --description "GitHub Actions OIDC - deploy da API recsys no EKS ${CLUSTER} (ns ${NAMESPACE})" \
    --assume-role-policy-document "$TRUST"
fi

aws iam put-role-policy --role-name "$ROLE_NAME" \
  --policy-name "eks-describe-${CLUSTER}" --policy-document "$POLICY"

# S3: dvc/ para o remote do DVC, recsys-api/ para o layout que o initContainer
# da API consome (publicado por scripts/publish_api_artifacts.sh).
S3_POLICY="$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucketForDvcAndApi",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::${BUCKET}"
    },
    {
      "Sid": "ReadWriteDvcAndApiObjects",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": [
        "arn:aws:s3:::${BUCKET}/dvc/*",
        "arn:aws:s3:::${BUCKET}/${API_PREFIX}/*"
      ]
    }
  ]
}
EOF
)"

aws iam put-role-policy --role-name "$ROLE_NAME" \
  --policy-name "dvc-s3-${BUCKET}" --policy-document "$S3_POLICY"

echo ">> Garantindo access entry no cluster $CLUSTER (escopo: ns/$NAMESPACE)..."
if ! aws eks describe-access-entry --cluster-name "$CLUSTER" --region "$REGION" \
     --principal-arn "$ROLE_ARN" >/dev/null 2>&1; then
  aws eks create-access-entry --cluster-name "$CLUSTER" --region "$REGION" \
    --principal-arn "$ROLE_ARN" --type STANDARD >/dev/null
fi

aws eks associate-access-policy --cluster-name "$CLUSTER" --region "$REGION" \
  --principal-arn "$ROLE_ARN" \
  --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSEditPolicy \
  --access-scope "type=namespace,namespaces=${NAMESPACE}" >/dev/null

echo
echo ">> Pronto. Configure no GitHub (Settings > Secrets and variables > Actions > Variables):"
echo "   AWS_ROLE_ARN = ${ROLE_ARN}"
echo
echo "   Ou via CLI: gh variable set AWS_ROLE_ARN --body ${ROLE_ARN}"
