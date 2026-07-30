# Pendências

Itens identificados e **ainda não corrigidos**, com referência de arquivo e
linha. Levantado em 2026-07-29, após o PR #19.

Referências cruzadas: [REQUIREMENTS.md](REQUIREMENTS.md) (requisitos do edital),
[../k8s/api/README.md](../k8s/api/README.md) (deploy).

## Contexto — o que já foi resolvido

Não repetir estas frentes; estão fechadas:

- Cluster EKS compartilhado `arcobridgegitops` em vez de provisionar cluster
  próprio; workflows/scripts de ciclo de vida removidos (PR #19)
- Role OIDC `github-actions-recsys-api` com privilégio mínimo — `eks:DescribeCluster`
  no cluster, S3 restrito ao prefixo `dvc/`, access entry escopada ao namespace
  `models` (PR #19)
- Remote DVC apontando para `s3://arcobridgegitops-models-177300752486/dvc`
  (antes era `/tmp/dvc-store`, que não sobrevivia a reboot) (PR #19)
- Dataset real do Instacart rastreado em `data/raw.dvc`; artefatos migrados do
  git para o cache DVC (#18)
- Gerador sintético não sobrescreve mais o ponteiro do dataset real (PR #19)

**Ação pendente de configuração** (não é código):

```bash
gh variable set AWS_ROLE_ARN --body arn:aws:iam::177300752486:role/github-actions-recsys-api
```

---

## A. Bloqueiam o deploy da API

Enquanto qualquer um destes existir, o `deploy-api.yml` falha no
`rollout status` e dispara o rollback.

- [ ] **A1. Probes apontam para rota inexistente.** `readinessProbe` e
      `livenessProbe` usam `path: /health`, mas o router tem prefixo `/health` e
      a rota é `/status` — o endpoint real é `/health/status`. O probe recebe 404
      e o pod nunca fica Ready.
      → `k8s/api/deployment.yaml:68` e `:74` vs `src/api/routes/health.py:8,11`
      → **Uma linha cada. Item mais barato e mais letal da lista.**

- [ ] **A2. Bucket placeholder.** `MODELS_BUCKET: REPLACE_WITH_YOUR_BUCKET` faz
      o initContainer falhar no `aws s3 sync`. O bucket correto é
      `arcobridgegitops-models-177300752486`, mas o prefixo `recsys-api/` ainda
      não existe nele — depende de rodar `scripts/publish_api_artifacts.sh`.
      → `k8s/api/deployment.yaml:38`

- [ ] **A3. Imagem placeholder.** `image: REGISTRY/recsys-api:latest`. O
      `deploy-api.yml` corrige via `kubectl set image`, mas o `apply -k` cria o
      Deployment com a imagem inválida antes disso.
      → `k8s/api/deployment.yaml:59`

- [ ] **A4. Validar o IRSA da SA `sa-model-s3`.** A SA existe no namespace
      `models` e aponta para
      `arn:aws:iam::177300752486:role/eksctl-arcobridgegitops-addon-iamserviceaccou-Role1-r7CANGNmMzxy`.
      Confirmar que essa role alcança o bucket/prefixo escolhido em A2.

---

## B. Pipeline e dados

- [ ] **B1. Gerar e publicar os artefatos com o dataset real.** O remote S3 tem
      apenas o `data/raw`; os 20 outputs do pipeline (`data/processed/*`,
      `models/*`, `MODEL_CARD.md`, `metrics_comparison.csv`) estão `missing`.
      Os artefatos e métricas atuais foram gerados com o dataset **sintético**.
      → `uv run dvc repro && uv run dvc push`
      → Atenção ao tempo: `cosine_similarity` 3000×3000 sobre ~131k usuários no
        item-CF, mais 4 trials de random search do NCF.

- [ ] **B2. `model-release.yml` pode estourar o timeout.** `timeout-minutes: 60`
      foi dimensionado para o sintético. Decidir entre aumentar o limite ou
      restringir o workflow a `workflow_dispatch`.
      → `.github/workflows/model-release.yml`

---

## C. Requisitos do edital em risco

- [ ] **C1. Modelo promovido ≠ modelo servido.** Três inconsistências na mesma
      cadeia:
      1. `evaluate` promove o melhor por recall, mas sempre sob o nome fixo
         `item_based_cf_recommender` — o `MODEL_CARD.md` chega a documentar
         "`item_based_cf_recommender` (pyfunc servindo **popularity**)".
      2. A API em modo `local` sempre instancia `item_based_cf`, ignorando o
         vencedor da avaliação.
      3. A resposta reporta `model_type` a partir do settings, então em modo
         `registry` ela afirma "item_based_cf" mesmo servindo outro modelo.
      → `src/config.py:51`, `src/api/config.py:34`,
        `src/api/routes/recommendations.py:52`
      → **Mais visível numa banca:** a contradição está escrita no model card.

- [ ] **C2. Comparação de modelos não é apples-to-apples.** `_prep` recorta
      `users[:min(3000, …)]` — subconjunto **não aleatório** (menores `user_idx`)
      — e só `user_based_cf` é avaliado nele, com ground truth próprio; os outros
      4 usam o split de teste inteiro. `coverage_at_k` fica ainda menos
      comparável (numerador de 3.000 usuários, denominador do catálogo cheio).
      As 5 linhas aparecem lado a lado no card sem ressalva.
      → `src/pipeline/evaluate.py:128`

- [ ] **C3. MODEL_CARD sem limitações e vieses.** O edital (requisito 4) pede
      "performance, limitações e possíveis vieses"; o template gera apenas
      tabela de métricas e decisão de promoção.
      → `src/pipeline/evaluate.py:264`

- [ ] **C4. Cobertura de testes em 31%, com 0% no núcleo de ML.** Sem nenhum
      teste: `src/evaluation/metrics.py`, `src/pipeline/*`,
      `src/models/training/*`, `src/serving/pyfunc.py`. As métricas que decidem
      a promoção do modelo não são testadas. Os 33 testes existentes cobrem
      apenas API e inferência.

- [ ] **C5. Deps incompletas no estágio `evaluate`.** Faltam
      `src/serving/pyfunc.py`, `src/evaluation/`, `configs/model_config.yaml` e
      `data/processed`. Alterar o `k` da avaliação ou o wrapper de serving não
      invalida o estágio — reprodutibilidade furada.
      → `dvc.yaml:45`

- [ ] **C6. "Validação cruzada estratificada" não acontece.** O
      `StratifiedKFold` é usado apenas para particionar usuários em
      train/val/test. Não há cross-validation: todo random search avalia em um
      único split de validação.
      → `src/pipeline/preprocess.py:145`

---

## D. Qualidade e consistência

- [ ] **D1. Seed com 5 fontes**, contra a regra de fonte única do `CLAUDE.md`:
      `src/config.py:17`, `src/models/training/ncf_train.py:22`,
      `src/api/config.py:44`, mais `42` hardcoded em
      `src/models/training/item_cf.py:38` e `user_cf.py:90` — estes dois
      ignoram o seed recebido (`item_cf.train` nem recebe o parâmetro).

- [ ] **D2. `max_top_k` nunca é aplicado.** Configurado como 100
      (`src/api/config.py:41`), mas o schema aceita `le=1000`
      (`src/api/schemas/request.py:17`). `default_top_k` também é morto — o
      schema hardcoda `default=10`.

- [ ] **D3. Strategy Pattern é código morto.** `src/preprocessing/`
      (`InteractionFilterStrategy`, `UserItemEncoderStrategy`) não é importado
      por `src/pipeline/preprocess.py`, que reimplementa a lógica. O padrão só
      é exercitado por testes e pelo notebook 02.

- [ ] **D4. Config declarada ≠ config real.** `configs/model_config.yaml:1`
      (`random_seed`) nunca é lido. `train_ratio` só é ecoado em
      `split_meta.json`: o split real vem de `_folds_for_ratio(0.15)` = 7 folds,
      ou seja 71,4/14,3/14,3 — não 70/15/15.

- [ ] **D5. Serving ineficiente.** `_UserCFScorer.__call__` refatia até 20k
      linhas esparsas a cada request (`src/serving/pyfunc.py:126`);
      `_PopularityScorer.__call__` reconstrói o vetor de scores a cada chamada.

- [ ] **D6. `inference_latency_ms` é enganoso.** `_timed`
      (`src/pipeline/evaluate.py:99`) mede o custo de um batch dividido pelo
      número de usuários; o card publica isso como latência de inferência, que
      não corresponde ao perfil por request da API.

- [ ] **D7. Docstring desatualizada.** `RegistryRecommender` cita
      `ItemBasedCFPyfunc`, classe que não existe mais (hoje `RecommenderPyfunc`).
      → `src/models/registry_recommender.py:21`

- [ ] **D8. Idiomas misturados.** `src/models/` e `src/pipeline/` em inglês,
      `src/api/` em português — inclusive dentro do mesmo arquivo
      (`src/models/inference.py:109`).

- [ ] **D9. API do MLflow deprecated.** `transition_model_version_stage`
      (`src/pipeline/evaluate.py:222-223`) foi descontinuada na 2.9 e removida
      na 3.x. Funciona pelo pin `mlflow<3`, mas trava o upgrade.

---

## Ordem sugerida

**Se a prioridade é a nota do Tech Challenge:** C3 → C5 → C4 → C1.
São requisitos explícitos do edital, baratos, e não dependem de AWS. O C1 vem
logo depois por ser a inconsistência mais visível em uma apresentação.

**Se a prioridade é ter a API no ar:** A1 → B1 → A2 → A3 → A4.
O A1 é uma linha; o B1 gera os artefatos que o A2 precisa publicar.
