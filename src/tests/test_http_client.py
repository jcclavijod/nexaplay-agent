"""Tests unitarios para NexaPlayHTTPClient.

Cubre: happy path GET/POST, idempotency-key determinística, retry con backoff,
no-retry en 4xx, timeout retry, SILENT_WRITE_FAILURE y falso positivo.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import httpx
import pytest
import respx

from src.skills.nexaplay_api.http_client import NexaPlayHTTPClient, _IDEMPOTENCY_NS

BASE_URL = "https://nexaplay.test"
ENDPOINT = "/services/42/config"
FULL_URL = BASE_URL + ENDPOINT


# ── Helpers de respuesta ─────────────────────────────────────────────────────

def _ok_get() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "success": True,
            "data": {
                "service_id": 42,
                "client_id": "MX-01",
                "country": "MX",
                "operational_limit": 92,
                "min_allowed": 95,
                "standard_value": 96,
                "max_transactions_per_second": 100,
                "business_rules": {},
                "metadata": {},
            },
        },
    )


def _ok_post(updated: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "success": True,
            "data": {
                "service_id": 42,
                "client_id": "MX-01",
                "country": "MX",
                "previous": {},
                "updated": updated,
                "timestamp": "2024-01-01T00:00:00Z",
                "version": 2,
            },
        },
    )


# ── Tests ────────────────────────────────────────────────────────────────────

@respx.mock
async def test_get_happy_path():
    """GET 200: success=True, data presente, sin reintentos."""
    respx.get(FULL_URL, params={"client_id": "MX-01", "country": "MX"}).mock(
        return_value=_ok_get()
    )
    async with NexaPlayHTTPClient(BASE_URL) as client:
        result = await client.call(
            ENDPOINT, "GET", "job-1", 1,
            params={"client_id": "MX-01", "country": "MX"},
        )

    assert result["success"] is True
    assert result["data"] is not None
    assert result["retries_used"] == 0


@respx.mock
async def test_post_includes_idempotency_key():
    """POST: el header Idempotency-Key es el UUID v5 esperado para (job_id, step)."""
    route = respx.post(FULL_URL).mock(
        return_value=_ok_post({"operational_limit": 96})
    )
    async with NexaPlayHTTPClient(BASE_URL) as client:
        await client.call(ENDPOINT, "POST", "job-42", 7, body={"operational_limit": 96})

    expected = str(uuid.uuid5(_IDEMPOTENCY_NS, "job-42-7"))
    actual = route.calls[0].request.headers["Idempotency-Key"]
    assert actual == expected


@respx.mock
async def test_post_same_step_reuses_idempotency_key():
    """Dos POST con mismo job_id+step producen idéntico Idempotency-Key."""
    route = respx.post(FULL_URL).mock(
        return_value=_ok_post({"operational_limit": 96})
    )
    async with NexaPlayHTTPClient(BASE_URL) as client:
        await client.call(ENDPOINT, "POST", "job-99", 3, body={"operational_limit": 96})
        await client.call(ENDPOINT, "POST", "job-99", 3, body={"operational_limit": 96})

    k1 = route.calls[0].request.headers["Idempotency-Key"]
    k2 = route.calls[1].request.headers["Idempotency-Key"]
    assert k1 == k2


@respx.mock
async def test_5xx_retries_three_times():
    """4 × 503 → 3 reintentos, success=False, error SERVER_ERROR."""
    route = respx.get(FULL_URL).mock(return_value=httpx.Response(503))
    async with NexaPlayHTTPClient(BASE_URL) as client:
        with patch("src.skills.nexaplay_api.http_client.asyncio.sleep"):
            result = await client.call(ENDPOINT, "GET", "job-1", 1)

    assert result["success"] is False
    assert "SERVER_ERROR" in result["error"]
    assert result["retries_used"] == 3
    assert route.call_count == 4


@respx.mock
async def test_5xx_then_success():
    """503 seguido de 200 → success=True, retries_used=1."""
    responses = iter([httpx.Response(503), _ok_get()])
    respx.get(FULL_URL).mock(side_effect=lambda req: next(responses))
    async with NexaPlayHTTPClient(BASE_URL) as client:
        with patch("src.skills.nexaplay_api.http_client.asyncio.sleep"):
            result = await client.call(ENDPOINT, "GET", "job-1", 1)

    assert result["success"] is True
    assert result["retries_used"] == 1


@respx.mock
async def test_4xx_no_retry():
    """400 → 1 solo intento, success=False, error VALIDATION_ERROR."""
    route = respx.get(FULL_URL).mock(return_value=httpx.Response(400))
    async with NexaPlayHTTPClient(BASE_URL) as client:
        result = await client.call(ENDPOINT, "GET", "job-1", 1)

    assert result["success"] is False
    assert "VALIDATION_ERROR" in result["error"]
    assert route.call_count == 1


@respx.mock
async def test_timeout_retries():
    """TimeoutException en los 4 intentos → retries_used=3, error TIMEOUT_ERROR."""
    def raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout", request=request)

    respx.get(FULL_URL).mock(side_effect=raise_timeout)
    async with NexaPlayHTTPClient(BASE_URL) as client:
        with patch("src.skills.nexaplay_api.http_client.asyncio.sleep"):
            result = await client.call(ENDPOINT, "GET", "job-1", 1)

    assert result["success"] is False
    assert "TIMEOUT_ERROR" in result["error"]
    assert result["retries_used"] == 3


async def test_backoff_progression(monkeypatch: pytest.MonkeyPatch):
    """asyncio.sleep se llama con 1.0, 2.0, 4.0 en orden tras 4 × 503."""
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(
        "src.skills.nexaplay_api.http_client.asyncio.sleep", fake_sleep
    )

    with respx.mock:
        respx.get(FULL_URL).mock(return_value=httpx.Response(503))
        async with NexaPlayHTTPClient(BASE_URL) as client:
            await client.call(ENDPOINT, "GET", "job-1", 1)

    assert sleep_calls == [1.0, 2.0, 4.0]


@respx.mock
async def test_silent_write_failure_detected():
    """POST 200 con updated.operational_limit≠enviado → success=False, SILENT_WRITE_FAILURE, sin retry."""
    route = respx.post(FULL_URL).mock(
        return_value=_ok_post({"operational_limit": 92})  # enviamos 96, API devuelve 92
    )
    async with NexaPlayHTTPClient(BASE_URL) as client:
        result = await client.call(
            ENDPOINT, "POST", "job-1", 1, body={"operational_limit": 96}
        )

    assert result["success"] is False
    assert "SILENT_WRITE_FAILURE" in result["error"]
    assert route.call_count == 1  # cero reintentos


@respx.mock
async def test_post_correct_update_no_false_positive():
    """POST 200 con updated coincidente → success=True, sin falso SILENT_WRITE_FAILURE."""
    respx.post(FULL_URL).mock(
        return_value=_ok_post({"operational_limit": 96})
    )
    async with NexaPlayHTTPClient(BASE_URL) as client:
        result = await client.call(
            ENDPOINT, "POST", "job-1", 1, body={"operational_limit": 96}
        )

    assert result["success"] is True
