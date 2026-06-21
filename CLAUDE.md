# CLAUDE.md

Leia este arquivo antes de qualquer ação.
Para detalhes, leia os arquivos referenciados em docs/.

<important>
- RANDOM_SEED = 42 em todo código. Nunca use random/numpy/torch sem fixar seed.
- Proibido usar print() em qualquer módulo de src/. Use structlog. Prints são permitidos em notebooks.
- Linting (ruff) deve passar sem erros antes de qualquer commit.
- Validação cruzada sempre estratificada (StratifiedKFold).
- Todo experimento MLflow deve registrar: params, métricas, dataset hash e artefatos.
</important>

## Visão geral

Sistema de recomendação de produtos para um e-commerce, baseado no comportamento de navegação dos usuários.

- **Dataset**: Dados de interações de e-commerce (instacart-online-grocery-basket-analysis-dataset {https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset?resource=download}) 
- **Modelo principal**: Rede neural (MLP ou embedding-based) treinada com PyTorch.

---

## Stack Tecnológico e Ferramentas

- **Linguagem & Dependências**: Python com gerenciamento via pyproject.toml usando uv.  
- **Modelagem**: PyTorch para a rede neural e Scikit-Learn para baselines e pré-processamento.
- **Rastreamento & Registro**: MLflow para tracking de experimentos e Model Registry.
- **Versionamento**: DVC para versionamento de dados e pipeline reprodutível.
- **Infraestrutura**: Docker com Dockerfile multi-stage e docker-compose.yml. 

---

## Regras de Arquitetura e Clean Code

- **Estrutura de Diretórios**: O projeto deve obrigatoriamente conter as pastas src/, tests/, data/, models/ e configs/.  
- **Qualidade de Código**: As funções devem ter no máximo 20 linhas.  
- **Tipagem e Documentação**: Uso obrigatório de type hints em todas as funções públicas e docstrings no padrão Google style.  
- **Linting e Hooks**: Configuração do Ruff sem erros e utilização de pre-commit hooks.  
- **Design Patterns**: Implementação de pelo menos um padrão de projeto, como Factory (para criar modelos), Strategy (para pré-processadores) ou Template Method.  
- **Ambiente Isolado**: Separação clara entre dependências de produção e desenvolvimento, com o lock file comitado e configurações externalizadas via .env.  

---

## Contexto detalhado

| Arquivo | Conteúdo |
|---------|----------|
| `docs/REQUIREMENTS.md` | Lista completa de requisitos obrigatórios do Tech Challenge 2 |
| `docs/NOTEBOOKS.md` | Estrutura e propósito dos notebooks (EDA, Baseline, Treinamento, Avaliação) |
| `docs/api-conventions.md` | Padrões e convenções para desenvolvimento da API REST (FastAPI) |
| `docs/COMMIT-CONVENTIONS.md` | Padrão Conventional Commits para histórico de commits |
| `docs/naming-conventions.md` | Convenções de código, seeds, logging, commits |
| `docs/design-pattern.md` | Convenções para os padrões de projeto |

---