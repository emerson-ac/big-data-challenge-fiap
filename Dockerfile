# Multi-stage build: builder resolves deps with uv; runtime carries venv + code.
# python:3.12-slim base, non-root user, healthcheck (Aulas Docker 1-3).
# Runtime keeps only inference-essential deps: training/viz packages (torch,
# matplotlib, seaborn) are resolved but NOT installed, shrinking the image
# (Aula 3 - Boas Praticas: "manter apenas runtime e componentes essenciais
# para inferencia" + "reducao de pacotes instalados").

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
# Slim deps set: skip training/viz-only packages (not imported at serve time).
ARG SERVING_NO_INSTALL="--no-install-package torch --no-install-package matplotlib --no-install-package seaborn"
# 1) Install dependencies only (cacheable layer, lock-pinned).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project ${SERVING_NO_INSTALL}
# 2) Copy code and install the project.
COPY . .
RUN uv sync --frozen --no-dev ${SERVING_NO_INSTALL}

FROM python:3.12-slim AS runtime
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app
USER appuser
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/status')" || exit 1
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
