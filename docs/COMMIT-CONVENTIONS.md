# Convenções de Commits - Conventional Commits 1.0.0

Documento que padroniza o histórico de commits do projeto seguindo a especificação **Conventional Commits 1.0.0**.

---

## Visão Geral

O Conventional Commits é uma especificação para adicionar significado aos históricos de commits. As commits são estruturadas de forma legível tanto para humanos quanto para máquinas, facilitando a rastreabilidade, geração automática de changelogs e versionamento semântico.

**Especificação oficial:** https://www.conventionalcommits.org/en/v1.0.0/

---

## Estrutura Básica

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Exemplo Completo

```
feat(recommendation): add user-based collaborative filtering

Implement Pearson correlation similarity metric for user
recommendation generation. Achieved 15% improvement in
recall@10 compared to baseline.

Closes #42
```

---

## Tipos de Commits

| Tipo | Descrição | SemVer | Exemplo |
|------|-----------|--------|---------|
| `feat` | Nova funcionalidade | Minor | `feat(api): add batch recommendations endpoint` |
| `fix` | Correção de bug | Patch | `fix(preprocessing): handle null values in RFM` |
| `refactor` | Reestruturação sem mudanças de comportamento | Patch | `refactor(model): extract preprocessing to service` |
| `perf` | Melhoria de performance | Patch | `perf(inference): optimize embedding lookup 2x faster` |
| `style` | Mudanças de formatação (sem alterar lógica) | - | `style: format code with ruff` |
| `test` | Adição ou atualização de testes | - | `test(baseline): add unit tests for knn recommender` |
| `docs` | Mudanças em documentação | - | `docs: update API conventions guide` |
| `chore` | Atualizações de dependências, build, CI/CD | - | `chore(deps): upgrade torch to 2.1.0` |
| `ci` | Mudanças em configuração CI/CD | - | `ci: add github actions workflow for tests` |
| `build` | Mudanças em build system ou Docker | - | `build: optimize dockerfile with multi-stage build` |

---

## Scope (Escopo)

O escopo é **opcional** mas **recomendado**. Deve indicar a área/módulo afetado:

```
feat(api)           # API REST
feat(model)         # Modelo de ML
feat(preprocessing) # Pré-processamento
feat(docs)          # Documentação
feat(docker)        # Containerização
feat(dvc)           # Pipeline DVC
feat(mlflow)        # Rastreamento MLflow
```

### Exemplos com Scope

```
feat(api): add batch recommendations endpoint
fix(preprocessing): handle null values correctly
refactor(model): extract loader to factory pattern
docs(api-conventions): add authentication section
```

---

## Descrição

A descrição deve:
- [ ] Usar imperative, present tense: "change" não "changed" ou "changes"
- [ ] Não começar com maiúscula
- [ ] Não terminar com ponto (.)
- [ ] Ser concisa e descritiva (máximo 50 caracteres idealmente)
- [ ] Explicar O QUE e POR QUE, não COMO

### Bons Exemplos

```
feat(model): implement mlp architecture with embeddings
fix(preprocessing): correct feature scaling order
refactor(api): extract validation logic to schemas
perf(inference): reduce prediction latency by 40%
```

### Exemplos Ruins

```
feat(model): Added the MLP model     # Maiúscula, passado
fix(preprocessing): Fixed bugs       # Muito vago
refactor: refactored code            # Sem scope, sem detalhe
```

---

## Body (Corpo)

O corpo é **opcional** mas **recomendado** para commits complexos.

Deve explicar:
- Por que essa mudança foi necessária
- Qual problema resolve
- Quais são as implicações

**Regras:**
- Separar da descrição por uma linha em branco
- Envolver a 72 caracteres
- Usar linguagem clara e explicativa

### Exemplo com Body

```
feat(model): implement cross-validation with stratification

Previously, we used standard KFold which could lead to
data leakage due to unbalanced class distribution.

Now using StratifiedKFold to ensure each fold maintains
the original class proportions, improving model reliability
and reducing variance across folds.

Also updated evaluation metrics to report stratified results.
```

---

## Footer (Rodapé)

O rodapé é **opcional** e contém referências a issues, breaking changes, etc.

### Formato

```
<token>: <value>
```

### Tokens Comuns

| Token | Descrição | Exemplo |
|-------|-----------|---------|
| `Closes` | Fecha uma issue | `Closes #42` |
| `Fixes` | Corrige uma issue | `Fixes #123` |
| `Refs` | Referencia uma issue | `Refs #456` |
| `BREAKING CHANGE` | Mudança que quebra compatibilidade | `BREAKING CHANGE: removed deprecated endpoint /v1/recommend` |

### Exemplos com Footer

```
fix(preprocessing): correct normalization range

The previous implementation used 0-1 range but should
use standard scores (mean=0, std=1).

Fixes #89
```

```
feat(api): add authentication via JWT tokens

Implement JWT-based authentication for all endpoints.

BREAKING CHANGE: requests now require Authorization header
with valid JWT token. Old API key authentication removed.

Closes #102
```

---

## Breaking Changes (Mudanças que Quebram)

Commits que introduzem mudanças incompatíveis devem incluir:

1. `BREAKING CHANGE:` no footer, OU
2. Adicionar `!` antes dos dois-pontos

### Exemplos

```
feat(api)!: rename endpoint from /recommend to /recommendations

Closes #50
```

Ou:

```
refactor(model)!: change input shape from (batch, features) to (batch, 1, features)

BREAKING CHANGE: model now expects 3D tensors instead of 2D.
Update all inference code accordingly.
```

---

## Exemplos Práticos

### Novo Recurso com Detalhes

```
feat(model): add attention mechanism to recommendation network

Implement multi-head attention layer to improve model's ability
to weight important user interaction features.

- Added AttentionLayer class with 8 heads
- Integrated into MLP architecture after embedding layer
- Achieved 8% NDCG improvement on validation set

Performance: inference time increased by 12ms (acceptable trade-off)

Closes #34
Refs #12
```

### Correção de Bug

```
fix(preprocessing): handle missing user_id gracefully

Previously crashed when encountering orders with NULL user_id.
Now filters out these records and logs warning.

Added unit tests for edge cases.

Fixes #67
```

### Melhoria de Performance

```
perf(inference): cache model embeddings in memory

Pre-compute and cache user/item embeddings in Redis to
reduce redundant forward passes during batch recommendations.

Results:
- Reduced P95 latency from 850ms to 180ms
- Enabled batch size of 1000 (previously 100)

Closes #88
```

### Refatoração com Design Pattern

```
refactor(api): extract recommendation logic to service layer

Move business logic from endpoints to service classes following
Strategy pattern. Makes code more testable and reusable.

Changes:
- Created RecommendationService class
- Created ModelInferenceService class
- Updated endpoints to use dependency injection

No behavior changes. All tests passing.
```

### Atualização de Dependências

```
chore(deps): upgrade pytorch to 2.1.0 and torch-geometric to 2.3

Updated to latest stable versions. Verified compatibility with
existing model checkpoints and inference code.

Testing:
- Ran full test suite (all passing)
- Retrained baseline models (no performance regression)
```

### Mudança em Documentação

```
docs(api-conventions): add authentication and rate limiting sections

Added comprehensive guide on JWT authentication, rate limiting
strategies, and error handling patterns.

Also updated examples to match latest API v2.0.0 responses.
```

---

## Integração com Pre-commit Hooks

### Instalação

O projeto já possui configuração em `.pre-commit-config.yaml` que valida commits:

```bash
uv run pre-commit install
```

### Validação Automática

O hook `conventional-pre-commit` valida cada commit antes de ser criado:

```bash
git commit -m "feat(api): add new endpoint"
# Validado automaticamente ✓

git commit -m "add new stuff"
# Erro: não segue Conventional Commits ✗
```

---

## Git Log Formatado

### Visualizar com Formatação

```bash
# Últimos 10 commits com tipo destacado
git log --oneline -10

# Ver commits por tipo
git log --grep="^feat:" --oneline

# Ver apenas fixes
git log --grep="^fix:" --oneline

# Ver commits com escopo
git log --grep="^[a-z]*\(api\):" --oneline
```

### Aliases Úteis

Adicionar ao `.gitconfig`:

```bash
git config --global alias.cl "log --oneline -10"
git config --global alias.cf "log --grep='^feat:' --oneline"
git config --global alias.cfix "log --grep='^fix:' --oneline"
```

---

## Checklist de Qualidade

Antes de fazer push:

- [ ] Título segue padrão `<type>(<scope>): <description>`
- [ ] Descrição em imperativo, presente tense
- [ ] Não tem maiúscula inicial na descrição
- [ ] Não termina com ponto
- [ ] Se complexo, tem corpo explicando POR QUE
- [ ] Se fecha issue, tem `Closes #XX` no footer
- [ ] Se quebra compatibilidade, tem `BREAKING CHANGE:`
- [ ] Mensagem não tem typos
- [ ] Relacionado a apenas uma mudança lógica
- [ ] Pre-commit hooks passaram (validação automática)

---

## Boas Práticas

### Faça Commits Pequenos e Lógicos

```
Bom:
commit 1: feat(model): add mlp architecture
commit 2: feat(model): implement training loop
commit 3: test(model): add unit tests for mlp

Ruim:
commit 1: feat(model): add mlp, training, preprocessing, tests (gigante)
```

### Um Assunto por Commit

```
Bom:
feat(api): add batch recommendations endpoint
fix(preprocessing): handle missing values

Ruim:
feat(api): add endpoint, fix preprocessing, update docs (múltiplos)
```

### Não Misture Tipos

```
Bom:
feat(model): implement attention layer
test(model): add attention layer tests

Ruim:
feat(model): implement attention layer and add tests (misturado)
```

### Use Verbos Imperativos

```
Bom:
feat: add recommendation service
fix: correct normalization bug
refactor: extract preprocessing logic

Ruim:
feat: added recommendation service
fix: corrected normalization bug
refactor: extracting preprocessing logic
```

---

## Geração Automática de Changelog

Com commits bem estruturados, é possível gerar changelog automaticamente:

```bash
# Ferramentas recomendadas:
# - semantic-release (Node.js)
# - commitizen (Node.js + Python)
# - python-semantic-release (Python)

# Exemplo com semantic-release:
npm run release
# Gera: CHANGELOG.md, tag de versão, publicação
```

---

## Referências

- [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)
- [Semantic Versioning 2.0.0](https://semver.org/)
- [Git Conventions](https://git-scm.com/docs/git-commit)
- [Commitizen](http://commitizen.github.io/cz-cli/)
- [Semantic Release](https://semantic-release.gitbook.io/)

---

## Integração com Projeto

Consulte também:
- [REQUIREMENTS.md](REQUIREMENTS.md) - Requisitos gerais de commits
- [naming-conventions.md](naming-conventions.md) - Convenções de código
- [design-pattern.md](design-pattern.md) - Padrões implementados
