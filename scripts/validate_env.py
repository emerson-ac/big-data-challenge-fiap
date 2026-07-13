"""Environment validation entrypoint (reproducibility, Aula 01).

Verifies that a fresh clone can run the project: Python version, required
packages, .env-backed settings, seed and config file.

Usage::

    uv run python scripts/validate_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import check_environment, get_settings  # noqa: E402

PASSED = "[PASS]"
FAILED = "[FAIL]"


def main() -> int:
    """Prints the validation report and returns the process exit code."""
    settings = get_settings()
    print(f"MLflow tracking URI : {settings.mlflow_tracking_uri}")
    print(f"MLflow experiment   : {settings.mlflow_experiment_name}")
    print(f"Random seed         : {settings.random_seed}")
    print()
    results = check_environment()
    _print_report(results)
    return 0 if all(passed for _, passed, _ in results) else 1


def _print_report(results: list[tuple[str, bool, str]]) -> None:
    """Formats the check results as an aligned report table."""
    name_width = max(len(name) for name, _, _ in results)
    for name, passed, detail in results:
        status = PASSED if passed else FAILED
        print(f"  {status} {name:<{name_width}}  {detail}")


if __name__ == "__main__":
    sys.exit(main())
