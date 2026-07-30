---
name: dvc-repro-and-publish
description: Roda o pipeline DVC completo (preprocess → train → evaluate) e publica os artefatos no S3 no layout consumido pela API. Pede o nome do bucket e valida autenticação AWS antes de executar.
---

## Passo 1 — Coletar configurações

**Pergunte ao usuário antes de continuar:**

1. **Nome do bucket S3** onde os artefatos serão publicados (ex: `meu-bucket-models`)
2. **Prefixo S3** (padrão: `recsys-api` — confirme ou peça para alterar)
3. **Região AWS** (padrão: `us-east-1` — confirme ou peça para alterar)

Aguarde as respostas antes de prosseguir.

## Passo 2 — Validar autenticação AWS

```bash
aws sts get-caller-identity 2>&1
```

Se retornar erro, oriente o usuário a configurar credenciais com `aws configure` ou `aws sso login` antes de continuar. **Não prossiga sem autenticação válida.**

## Passo 3 — Verificar dados raw

```bash
ls -lh data/raw/ 2>/dev/null | head -10 || echo "Diretório data/raw/ não encontrado"
```

Se `data/raw/` estiver vazio ou ausente, informe o usuário e sugira:
- Baixar o dataset do Kaggle e colocar em `data/raw/`
- Ou gerar dados sintéticos: `uv run python scripts/gen_synthetic_data.py`

**Não prossiga sem dados raw.**

## Passo 4 — Rodar o pipeline DVC

```bash
uv run dvc repro 2>&1
```

Este passo pode levar vários minutos (treino de 5 modelos). Informe o usuário.

Se falhar, mostre o erro completo e **não prossiga** para o passo de publicação.

## Passo 5 — Verificar artefatos gerados

```bash
ls -lh models/item_based_cf/item_similarity.npz \
       models/baseline_popularity/ranking.pkl \
       data/processed/vocabularies.pkl \
       data/processed/interactions_prior.npz 2>&1
```

Todos os 4 arquivos devem existir. Se algum estiver ausente, informe o erro.

## Passo 6 — Publicar no S3

Use o bucket e prefixo informados no Passo 1:

```bash
BUCKET="<BUCKET_INFORMADO_PELO_USUARIO>"
PREFIX="<PREFIXO_INFORMADO_PELO_USUARIO>"
REGION="<REGIAO_INFORMADA_PELO_USUARIO>"

echo "Publicando em s3://$BUCKET/$PREFIX/ ..."

aws s3 cp models/item_based_cf/item_similarity.npz \
  "s3://$BUCKET/$PREFIX/models/item_based_cf/item_similarity.npz" --region "$REGION"

aws s3 cp models/baseline_popularity/ranking.pkl \
  "s3://$BUCKET/$PREFIX/models/baseline_popularity/ranking.pkl" --region "$REGION"

aws s3 cp data/processed/vocabularies.pkl \
  "s3://$BUCKET/$PREFIX/data/processed/vocabularies.pkl" --region "$REGION"

aws s3 cp data/processed/interactions_prior.npz \
  "s3://$BUCKET/$PREFIX/data/processed/interactions_prior.npz" --region "$REGION"

echo "Verificando arquivos publicados:"
aws s3 ls "s3://$BUCKET/$PREFIX/" --recursive --region "$REGION"
```

## Passo 7 — Confirmar sucesso

Liste os arquivos no S3 e confirme que os 4 artefatos estão presentes com tamanhos esperados.

Se tudo estiver correto, informe:
- ✅ Pipeline executado com sucesso
- ✅ Artefatos publicados em `s3://<bucket>/<prefixo>/`
- Próximo passo sugerido: reiniciar os pods para carregar os novos artefatos com `kubectl rollout restart deployment/recsys-api -n models`
