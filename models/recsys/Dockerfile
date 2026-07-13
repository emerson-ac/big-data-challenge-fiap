# syntax=docker/dockerfile:1
# Imagem do predictor KServe customizado (recomendador).
# Build a partir da RAIZ do repositório:
#   docker build -f models/recsys/Dockerfile -t <repo>/recsys:latest .
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=0 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
# Projeto + runtime do KServe (não faz parte das deps de produção da API).
RUN uv sync --frozen --no-dev && uv pip install "kserve>=0.14,<0.15"

# KServe custom predictor escuta em 8080 por padrão.
EXPOSE 8080
ENTRYPOINT ["python", "models/recsys/model_server.py", "--model_name", "recsys"]
