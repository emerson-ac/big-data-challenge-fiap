"""Exceções customizadas da API de recomendação."""

from typing import Any


class APIException(Exception):
    """Exceção base da API.

    Args:
        error_code: Código de erro interno, estável entre versões.
        message: Mensagem descritiva, segura para exibir ao cliente.
        status_code: Código HTTP correspondente.
        details: Contexto adicional do erro (opcional).
    """

    def __init__(
        self,
        error_code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotLoadedError(APIException):
    """Erro quando o modelo de recomendação não pode ser carregado."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            error_code="MODEL_NOT_LOADED",
            message="Modelo de recomendação não disponível",
            status_code=503,
            details={"reason": reason},
        )
