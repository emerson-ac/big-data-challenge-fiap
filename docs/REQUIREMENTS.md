# Tech Challenge 2 - Requisitos Obrigatórios

## 1. Objetivo e Dados

- [ ] **Problema:** Desenvolver um sistema de recomendação de produtos para um e-commerce.
- [ ] **Dataset:** Utilizar dados de interações com no mínimo 10.000 interações user-item.
- [ ] **Fonte:** Instacart, RetailRocket, MovieLens ou equivalente.

---

## 2. Estrutura de Código e Clean Code

- [ ] **Diretórios:** Presença obrigatória de `src/`, `tests/`, `data/`, `models/`, `configs/`.
- [ ] **Qualidade de Funções:** Máximo 20 linhas por função.
- [ ] **Princípios SOLID:** Aplicados na arquitetura do código.
- [ ] **Design Patterns:** Implementar no mínimo 1 padrão (Factory, Strategy ou Template Method).
- [ ] **Type Hints:** Obrigatório em todas as funções públicas.
- [ ] **Docstrings:** Padrão Google Style em todas as funções públicas.
- [ ] **Linting:** Ruff configurado e sem erros.
- [ ] **Pre-commit Hooks:** Configurados e funcionais.
- [ ] **Commits Semânticos:** Histórico seguindo padrão Conventional Commits.

---

## 3. Gerenciamento de Ambiente e Dependências

- [ ] **Gerenciador:** Utilizar `pyproject.toml` com Poetry ou uv.
- [ ] **Separação de Dependências:** Distinguir produção (pytorch, sklearn, mlflow, dvc) e desenvolvimento (pytest, ruff).
- [ ] **Lock File:** Arquivo de trava comitado no repositório.
- [ ] **Validação de Ambiente:** Script `scripts/validate_env.py` implementado.
- [ ] **Externalização de Configuração:** Parâmetros em arquivo `.env`.
- [ ] **Arquivo de Exemplo:** `.env.example` criado.
- [ ] **Validação com Pydantic:** Settings validadas via Pydantic.

---

## 4. Modelagem e Ferramentas de ML

- [ ] **Modelo Principal:** Rede neural (MLP ou baseada em embeddings) implementada em PyTorch.
- [ ] **Baselines:** Comparação com modelos Scikit-Learn (ex: Nearest Neighbors, Matriz de Co-ocorrência).
- [ ] **Métricas de Avaliação:** No mínimo 4 métricas distintas (ex: Precision@K, Recall@K, NDCG, MAP).
- [ ] **Model Card:** Documento detalhando performance, limitações e possíveis vieses.

---

## 5. Infraestrutura e MLOps

### Docker e Containerização

- [ ] **Dockerfile:** Construído com múltiplos estágios (otimização de tamanho).
- [ ] **docker-compose.yml:** Arquivo para orquestração de serviços.

### DVC (Versionamento de Dados e Pipeline)

- [ ] **Configuração de Remote:** DVC configurado com storage remoto.
- [ ] **dvc.yaml:** Pipeline reprodutível com no mínimo 3 estágios.
  - [ ] Estágio 1: Pré-processamento (preprocess)
  - [ ] Estágio 2: Treinamento (train)
  - [ ] Estágio 3: Avaliação (evaluate)
- [ ] **dvc.lock:** Arquivo de trava gerado e commitado.

### MLflow

- [ ] **Tracking:** Rastreamento de parâmetros, métricas e artefatos.
- [ ] **Runs Mínimas:** No mínimo 3 execuções rastreadas.
- [ ] **Model Registry:** Melhor modelo registrado.
- [ ] **Ciclo de Vida:** Modelo promovido por estágios (Staging → Production).

---

## 6. Entregáveis Finais

- [ ] **Repositório GitHub:** 
  - [ ] Código completo versionado.
  - [ ] `.gitignore` configurado.
  - [ ] `.dockerignore` configurado.
  - [ ] Pipeline DVC funcional e reprodutível.
  - [ ] README detalhado com instruções de uso.
  
- [ ] **Vídeo STAR:**
  - [ ] Duração: Até 5 minutos.
  - [ ] Método: Situação → Tarefa → Ação → Resultado.
  - [ ] Conteúdo: Apresentação clara do problema e solução.

---

## Referências

Para detalhes específicos sobre convenções e estrutura, consulte:
- [NOTEBOOKS.md](NOTEBOOKS.md) - Estrutura detalhada dos notebooks do projeto (EDA, Baseline, Treinamento, Avaliação).
- [COMMIT-CONVENTIONS.md](COMMIT-CONVENTIONS.md) - Padrão Conventional Commits para estruturação de commits.
- [api-conventions.md](api-conventions.md) - Padrões e convenções para desenvolvimento da API REST (FastAPI).
- [naming-conventions.md](naming-conventions.md) - Convenções de código, seeds, logging e commits.
- [design-pattern.md](design-pattern.md) - Implementação de padrões de projeto.

---

## Notas Importantes

1. O descumprimento de requisitos estruturais (pastas, limite de linhas, design patterns) impacta diretamente na nota de avaliação.
2. Toda experimentação deve ser rastreada no MLflow com identificação clara de dataset e hiperparâmetros.
3. RANDOM_SEED = 42 deve ser fixado em todo código (numpy, torch, sklearn).
4. Validação cruzada deve ser estratificada (StratifiedKFold).
5. Proibido usar print() em código de produção; utilizar structlog para logging.
