"""Configuração de logging estruturado (structlog) da API."""

import logging

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configura structlog para logging estruturado em toda a API.

    Args:
        log_level: Nível mínimo de log (ex.: "INFO", "DEBUG").
    """
    logging.basicConfig(level=log_level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger() -> structlog.BoundLogger:
    """Obtém um logger estruturado configurado para a aplicação.

    Returns:
        Logger pronto para uso com kwargs estruturados.
    """
    return structlog.get_logger()
