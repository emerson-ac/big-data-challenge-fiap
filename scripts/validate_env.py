"""Valida o ambiente do projeto: versão de Python, dependências e settings.

Uso:
    uv run python scripts/validate_env.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "mlflow",
    "dvc",
    "fastapi",
    "structlog",
    "pydantic_settings",
]
MIN_PYTHON = (3, 12)


def check_python() -> list[str]:
    """Retorna erros se a versão de Python for inferior à mínima suportada."""
    if sys.version_info < MIN_PYTHON:
        current = ".".join(map(str, sys.version_info[:3]))
        return [f"Python {current} < {MIN_PYTHON[0]}.{MIN_PYTHON[1]} exigido"]
    return []


def check_packages() -> list[str]:
    """Retorna a lista de pacotes obrigatórios que não puderam ser importados."""
    errors = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            errors.append(f"pacote ausente: {package}")
    return errors


def check_directories() -> list[str]:
    """Retorna erros se diretórios estruturais obrigatórios não existirem."""
    settings = get_settings()
    required = [settings.config_path, settings.models_dir]
    return [f"caminho ausente: {p}" for p in required if not p.exists()]


def main() -> int:
    """Executa todas as checagens e imprime o resultado. Retorna o exit code."""
    errors = check_python() + check_packages() + check_directories()
    if errors:
        for error in errors:
            print(f"[FALHA] {error}")
        return 1
    print("[OK] Ambiente válido: Python, dependências e estrutura conferidos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
