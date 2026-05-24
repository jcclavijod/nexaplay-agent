"""Cliente HTTP de bajo nivel para la API de NexaPlay.

Gestiona retry con backoff progresivo, idempotency-key determinística para POST,
detección de SILENT_WRITE_FAILURE y logging estructurado JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Literal

import httpx

from .errors import APICallError, ErrorCode, classify_http_status, should_retry

# ── Namespace fijo para UUID v5 determinístico ──────────────────────────────
_IDEMPOTENCY_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# ── Delays de backoff en segundos (intento 0→1, 1→2, 2→3) ──────────────────
_BACKOFF_SCHEDULE = (1.0, 2.0, 4.0)


# ── Logging ────────────────────────────────────────────────────
class _JSONFormatter(logging.Formatter):
    """Formateador de logging que serializa cada registro como JSON de una línea."""

    def format(self, record: logging.LogRecord) -> str:
        """Serializa *record* como JSON incluyendo timestamp, level, logger y mensaje."""
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
        }
        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["message"] = record.getMessage()
        return json.dumps(payload, ensure_ascii=False)


def _get_logger() -> logging.Logger:
    """Crea o reutiliza el logger del módulo con el formateador JSON configurado."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        level_name = os.getenv("LOG_LEVEL", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
        logger.setLevel(level)
        logger.propagate = False
    return logger


_log = _get_logger()


# ── Cliente HTTP ─────────────────────────────────────────────────────


class NexaPlayHTTPClient:
    """Cliente async para la API de NexaPlay.

    Encapsula retry con backoff progresivo, idempotency-key determinística para
    POST y detección de escrituras silenciosamente fallidas (SILENT_WRITE_FAILURE).
    Compatible con ``async with`` para gestión automática del ciclo de vida.

    Ejemplo::

        async with NexaPlayHTTPClient(base_url="https://api.nexaplay.com") as client:
            result = await client.call("/services/42/config", "GET", job_id="abc", step=1)
    """

    def __init__(
        self,
        base_url: str,
        timeout_sec: float = 5.0,
        max_retries: int = 3,
    ) -> None:
        """Inicializa el cliente HTTP.

        Args:
            base_url: URL base de la API de NexaPlay (sin barra final).
            timeout_sec: Timeout por petición en segundos.
            max_retries: Número máximo de reintentos ante errores retriables.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_sec),
        )

    # ── Ciclo de vida ─────────────────────────────────────────────────────

    async def close(self) -> None:
        """Cierra el cliente HTTP subyacente."""
        await self._client.aclose()

    async def __aenter__(self) -> "NexaPlayHTTPClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


    # ── Método principal ─────────────────────────────────────────────────────
    async def call(
        self,
        endpoint: str,
        method: Literal["GET", "POST"],
        job_id: str,
        step: int,
        params: dict | None = None,
        body: dict | None = None,
    ) -> dict[str, Any]:
        """Ejecuta una llamada HTTP contra NexaPlay con retry y clasificación de errores.

        Para POST genera una Idempotency-Key determinística a partir de ``job_id``
        y ``step``, garantizando exactamente una escritura incluso ante reintentos.
        Los errores no-retriables (4xx) y SILENT_WRITE_FAILURE se devuelven
        serializados en el dict de retorno sin lanzar excepciones.

        Args:
            endpoint: Path relativo del endpoint (ej. ``"/services/42/config"``).
            method: Método HTTP — ``"GET"`` o ``"POST"``.
            job_id: Identificador del job, parte de la Idempotency-Key.
            step: Número de paso dentro del job, parte de la Idempotency-Key.
            params: Query params opcionales para la petición.
            body: Body JSON para POST. Ignorado en GET.

        Returns:
            Dict con las keys ``success``, ``data``, ``error``, ``retries_used``,
            ``status_code`` y ``duration_ms``.
        """
        idempotency_key: str | None = None
        if method == "POST":
            idempotency_key = str(uuid.uuid5(_IDEMPOTENCY_NS, f"{job_id}-{step}"))

        last_error: APICallError | None = None
        retries_used = 0

        for attempt in range(self._max_retries + 1):
            t0 = time.monotonic()
            status_code: int | None = None

            try:
                response = await self._send(
                    endpoint=endpoint,
                    method=method,
                    params=params,
                    body=body,
                    idempotency_key=idempotency_key,
                )
                status_code = response.status_code
                duration_ms = int((time.monotonic() - t0) * 1000)

                self._log_attempt(
                    job_id=job_id,
                    step=step,
                    endpoint=endpoint,
                    method=method,
                    attempt=attempt,
                    status_code=status_code,
                    duration_ms=duration_ms,
                )

                if response.is_success:
                    return self._handle_success_response(
                        response=response,
                        method=method,
                        body=body,
                        retries_used=retries_used,
                        status_code=status_code,
                        duration_ms=duration_ms,
                    )

                # Respuesta no-2xx
                last_error = self._build_http_error(
                    status_code=status_code,
                    method=method,
                    endpoint=endpoint,
                    retries_used=retries_used,
                )

            except httpx.TimeoutException as exc:
                duration_ms = self._duration_ms(t0)
                last_error = self._build_timeout_error(
                    exc=exc,
                    retries_used=retries_used,
                )

                self._log_error(
                    job_id=job_id,
                    step=step,
                    endpoint=endpoint,
                    method=method,
                    attempt=attempt,
                    duration_ms=duration_ms,
                    error="TIMEOUT_ERROR",
                )

            except (httpx.ConnectError, httpx.NetworkError) as exc:
                duration_ms = self._duration_ms(t0)
                last_error = self._build_network_error(
                    exc=exc,
                    retries_used=retries_used,
                )

                self._log_error(
                    job_id=job_id,
                    step=step,
                    endpoint=endpoint,
                    method=method,
                    attempt=attempt,
                    duration_ms=duration_ms,
                    error="NETWORK_ERROR",
                )

            # ── Decidir si reintentar ─────────────────────────────────────
            if self._should_retry(last_error, attempt):
                retries_used += 1
                delay = self._get_backoff_delay(attempt)

                self._log_retry(
                    job_id=job_id,
                    step=step,
                    endpoint=endpoint,
                    method=method,
                    attempt=attempt,
                    delay=delay,
                )

                await asyncio.sleep(delay)
                continue

            # Error no-retriable o agotados los reintentos
            break

        # Aquí se llega solo por errores; serializar y retornar
        assert last_error is not None
        return self._error_response(
            error=last_error,
            retries_used=retries_used,
        )



    # ── Response handlers ─────────────────────────────────────────────────────


    def _handle_success_response(
        self,
        response: httpx.Response,
        method: str,
        body: dict | None,
        retries_used: int,
        status_code: int,
        duration_ms: int,
    ) -> dict[str, Any]:
        """Procesa una respuesta 2xx y verifica que no sea un SILENT_WRITE_FAILURE.

        Args:
            response: Respuesta HTTP exitosa de httpx.
            method: Método HTTP de la petición original.
            body: Body enviado en el POST, o ``None`` para GET.
            retries_used: Número de reintentos consumidos hasta esta respuesta.
            status_code: Código HTTP de la respuesta.
            duration_ms: Duración de la petición en milisegundos.

        Returns:
            Dict de resultado con ``success=True``, o con ``success=False`` si
            se detecta un SILENT_WRITE_FAILURE.
        """
        data = response.json()

        if method == "POST" and body:
            try:
                _check_silent_write_failure(body, data)

            except APICallError as exc:
                return self._error_response(
                    error=exc,
                    retries_used=retries_used,
                    status_code=status_code,
                    duration_ms=duration_ms,
                )

        return {
            "success": True,
            "data": data.get("data"),
            "error": None,
            "retries_used": retries_used,
            "status_code": status_code,
            "duration_ms": duration_ms,
        }
    

    # ── Transport ─────────────────────────────────────────────────────

    async def _send(
        self,
        endpoint: str,
        method: str,
        params: dict | None,
        body: dict | None,
        idempotency_key: str | None,
    ) -> httpx.Response:
        """Despacha la petición HTTP sin lógica de retry.

        Args:
            endpoint: Path relativo del endpoint.
            method: Método HTTP (``"GET"`` o ``"POST"``).
            params: Query params opcionales.
            body: Body JSON para POST.
            idempotency_key: Valor del header ``Idempotency-Key``, o ``None``.

        Returns:
            Respuesta HTTP de httpx sin procesar.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        if method == "GET":
            return await self._client.get(endpoint, params=params, headers=headers)
        return await self._client.post(
            endpoint, params=params, json=body, headers=headers
        )


    # ── Retry helpers ─────────────────────────────────────────────────────

    def _should_retry(
        self,
        error: APICallError | None,
        attempt: int,
    ) -> bool:
        """Retorna ``True`` si el error es retriable y quedan intentos disponibles."""
        return (
            error is not None
            and should_retry(error.code)
            and attempt < self._max_retries
        )

    @staticmethod
    def _get_backoff_delay(attempt: int) -> float:
        """Retorna el delay de backoff en segundos para el intento dado."""
        return _BACKOFF_SCHEDULE[
            min(attempt, len(_BACKOFF_SCHEDULE) - 1)
        ]
    

    # ── Response builders ─────────────────────────────────────────────────────

    @staticmethod
    def _error_response(
        error: APICallError,
        retries_used: int,
        status_code: int | None = None,
        duration_ms: int = 0,
    ) -> dict[str, Any]:
        """Serializa un :class:`~errors.APICallError` como dict de resultado con ``success=False``."""
        return {
            "success": False,
            "data": None,
            "error": f"{error.code.value}: {error.message}",
            "retries_used": retries_used,
            "status_code": status_code or error.status_code,
            "duration_ms": duration_ms,
        }
    

    # ── Error builders ─────────────────────────────────────────────────────

    @staticmethod
    def _build_http_error(
        status_code: int,
        method: str,
        endpoint: str,
        retries_used: int,
    ) -> APICallError:
        """Construye un :class:`~errors.APICallError` a partir de un código HTTP no-2xx."""
        return APICallError(
            code=classify_http_status(status_code),
            message=f"HTTP {status_code} en {method} {endpoint}",
            status_code=status_code,
            retries_used=retries_used,
        )

    @staticmethod
    def _build_timeout_error(
        exc: Exception,
        retries_used: int,
    ) -> APICallError:
        """Construye un :class:`~errors.APICallError` de tipo ``TIMEOUT_ERROR``."""
        return APICallError(
            code=ErrorCode.TIMEOUT_ERROR,
            message=str(exc),
            status_code=None,
            retries_used=retries_used,
        )

    @staticmethod
    def _build_network_error(
        exc: Exception,
        retries_used: int,
    ) -> APICallError:
        """Construye un :class:`~errors.APICallError` de tipo ``NETWORK_ERROR``."""
        return APICallError(
            code=ErrorCode.NETWORK_ERROR,
            message=str(exc),
            status_code=None,
            retries_used=retries_used,
        )

    # ── Logging ─────────────────────────────────────────────────────

    def _log_attempt(
        self,
        job_id: str,
        step: int,
        endpoint: str,
        method: str,
        attempt: int,
        status_code: int | None,
        duration_ms: int,
    ) -> None:
        """Registra el resultado de un intento HTTP como JSON estructurado."""
        _log.info(
            {
                "job_id": job_id,
                "step": step,
                "endpoint": endpoint,
                "method": method,
                "attempt_number": attempt + 1,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "retries_remaining": self._max_retries - attempt,
            }
        )

    def _log_retry(
        self,
        job_id: str,
        step: int,
        endpoint: str,
        method: str,
        attempt: int,
        delay: float,
    ) -> None:
        """Registra que se va a reintentar la petición tras un error retriable."""
        _log.info(
            {
                "job_id": job_id,
                "step": step,
                "endpoint": endpoint,
                "method": method,
                "attempt_number": attempt + 1,
                "backoff_seconds": delay,
                "retries_remaining": self._max_retries - attempt - 1,
                "message": "reintentando tras error retriable",
            }
        )

    def _log_error(
        self,
        job_id: str,
        step: int,
        endpoint: str,
        method: str,
        attempt: int,
        duration_ms: int,
        error: str,
    ) -> None:
        """Registra un error de red o timeout como advertencia JSON estructurada."""
        _log.warning(
            {
                "job_id": job_id,
                "step": step,
                "endpoint": endpoint,
                "method": method,
                "attempt_number": attempt + 1,
                "duration_ms": duration_ms,
                "retries_remaining": self._max_retries - attempt,
                "error": error,
            }
        )


    # ── Utilds ─────────────────────────────────────────────────────

    @staticmethod
    def _duration_ms(t0: float) -> int:
        """Retorna el tiempo transcurrido desde *t0* en milisegundos."""
        return int((time.monotonic() - t0) * 1000)

    @staticmethod
    def _build_idempotency_key(
        method: str,
        job_id: str,
        step: int,
    ) -> str | None:
        """Genera la Idempotency-Key determinística para POST, o ``None`` para GET.

        Args:
            method: Método HTTP de la petición.
            job_id: Identificador del job.
            step: Número de paso del plan.

        Returns:
            UUID v5 como string para ``"POST"``, ``None`` para cualquier otro método.
        """
        if method != "POST":
            return None

        return str(
            uuid.uuid5(
                _IDEMPOTENCY_NS,
                f"{job_id}-{step}",
            )
        )

def _check_silent_write_failure(
    body: dict[str, Any], response_data: dict[str, Any]
) -> None:
    """Verifica que los campos del body POST estén reflejados en la respuesta.

    Compara cada campo de *body* contra ``response_data["data"]["updated"]``.

    Args:
        body: Dict enviado como body del POST.
        response_data: JSON completo de la respuesta del servidor.

    Raises:
        APICallError: Con código ``SILENT_WRITE_FAILURE`` si algún campo difiere
            entre lo enviado y lo confirmado en la respuesta.
    """
    updated: dict[str, Any] = response_data.get("data", {}).get("updated") or {}
    mismatches: dict[str, dict] = {}
    for field, sent_value in body.items():
        actual = updated.get(field)
        if actual != sent_value:
            mismatches[field] = {"sent": sent_value, "got": actual}

    if mismatches:
        raise APICallError(
            code=ErrorCode.SILENT_WRITE_FAILURE,
            message=(
                "La API devolvió 2xx pero los campos actualizados no coinciden "
                f"con el body enviado: {mismatches}"
            ),
            status_code=None,
            retries_used=0,
            details={"mismatches": mismatches},
        )
