"""Tipos de error y helpers de clasificación para el cliente HTTP de NexaPlay."""
from __future__ import annotations

from enum import Enum


class ErrorCode(Enum):
    """Códigos de error canónicos del cliente HTTP de NexaPlay."""

    NETWORK_ERROR = "NETWORK_ERROR"
    SERVER_ERROR = "SERVER_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    SILENT_WRITE_FAILURE = "SILENT_WRITE_FAILURE"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"


class APICallError(Exception):
    """Excepción que encapsula cualquier error del cliente HTTP de NexaPlay.

    Transporta el :class:`ErrorCode` canónico, el mensaje legible, el código
    HTTP opcional y los reintentos consumidos hasta el fallo.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int | None = None,
        retries_used: int = 0,
        details: dict | None = None,
    ) -> None:
        """Inicializa el error con su clasificación y contexto.

        Args:
            code: Código de error canónico que clasifica la causa del fallo.
            message: Descripción legible del error.
            status_code: Código HTTP de la respuesta, o ``None`` si no aplica.
            retries_used: Número de reintentos consumidos antes de este fallo.
            details: Información adicional estructurada (ej. campos con mismatch).
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retries_used = retries_used
        self.details = details


def classify_http_status(status: int) -> ErrorCode:
    """Mapea un código HTTP a su :class:`ErrorCode` canónico.

    Args:
        status: Código HTTP de la respuesta (ej. 404, 503).

    Returns:
        ``VALIDATION_ERROR`` para 4xx, ``SERVER_ERROR`` para 5xx.

    Raises:
        ValueError: Si *status* es 2xx (los éxitos no son errores) o no está
            contemplado en ningún rango conocido.
    """
    if 200 <= status < 300:
        raise ValueError(f"HTTP {status} is a success status, not an error")
    if 400 <= status < 500:
        return ErrorCode.VALIDATION_ERROR
    if 500 <= status < 600:
        return ErrorCode.SERVER_ERROR
    raise ValueError(f"Unhandled HTTP status: {status}")


def should_retry(code: ErrorCode) -> bool:
    """Retorna ``True`` si la clase de error es segura para reintentar con backoff.

    Args:
        code: Código de error a evaluar.

    Returns:
        ``True`` para ``NETWORK_ERROR``, ``SERVER_ERROR`` y ``TIMEOUT_ERROR``;
        ``False`` para ``VALIDATION_ERROR`` y ``SILENT_WRITE_FAILURE``.
    """
    return code in {ErrorCode.NETWORK_ERROR, ErrorCode.SERVER_ERROR, ErrorCode.TIMEOUT_ERROR}
