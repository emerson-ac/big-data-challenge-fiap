# syntax=docker/dockerfile:1
# Build multi-stage: estágio "builder" resolve dependências com uv;
# estágio "runtime" carrega apenas o venv + código (imagem menor).

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
# 1) Instala só as dependências (camada cacheável) a partir do lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# 2) Copia o código e instala o projeto.
COPY . .
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime
WORKDIR /app
# Copia o ambiente virtual e o código já preparados.
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
EXPOSE 8000
# Servidor de produção da API de recomendação.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
