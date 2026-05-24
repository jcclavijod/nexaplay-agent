"""Tests para ContextManager (ADR sección 5)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.context_manager import ContextManager, Observation


def _make_obs(step_id: int) -> Observation:
    return Observation(
        step_id=step_id,
        tool_name="nexaplay_api_call",
        arguments={"service_id": "42", "client_id": "MX-01"},
        result={"success": True, "data": {"operational_limit": 92}},
        tokens_estimated=50,
    )


def _mock_summarize_response(text: str) -> MagicMock:
    """Construye un mock de respuesta de anthropic.messages.create."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


# ── add_observation: ventana deslizante ───────────────────────────────────────


@pytest.mark.asyncio
async def test_add_five_observations_yields_two_summarized_three_active() -> None:
    """Añadir 5 obs con max_recent=3 → 2 resumidas, 3 activas."""
    cm = ContextManager(max_recent_observations=3)

    call_count = 0

    async def fake_summarize(obs: Observation) -> str:
        nonlocal call_count
        call_count += 1
        return f"Resumen paso {obs.step_id}"

    cm._summarize = fake_summarize  # type: ignore[method-assign]

    for i in range(1, 6):
        await cm.add_observation(_make_obs(i))

    assert len(cm.observations) == 3
    assert len(cm.summarized_observations) == 2
    assert call_count == 2
    report = cm.get_budget_report()
    assert report["active_count"] == 3
    assert report["summarized_count"] == 2
    assert report["total_observations"] == 5


# ── _summarize: llama a Claude con modelo de resumen ─────────────────────────


@pytest.mark.asyncio
async def test_summarize_calls_claude_and_returns_string() -> None:
    """_summarize debe llamar a la API de Claude y retornar un string."""
    cm = ContextManager()
    obs = _make_obs(1)

    mock_response = _mock_summarize_response(
        "Tool nexaplay_api_call consultó servicio 42 (MX-01): success, operational_limit=92."
    )

    with patch.object(cm._client.messages, "create", new=AsyncMock(return_value=mock_response)):
        result = await cm._summarize(obs)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "nexaplay_api_call" in result or "92" in result


# ── estimate_tokens: retorna número razonable ─────────────────────────────────


def test_estimate_tokens_reasonable_count() -> None:
    """Un texto corto conocido debe dar un conteo de tokens coherente."""
    cm = ContextManager()
    text = "El agente actualizó el operational_limit de 92 a 96 para MX-01."
    tokens = cm.estimate_tokens(text)
    # Rango heurístico: ~1 token por ~4 caracteres; texto de ~65 chars → 12–25 tokens
    assert 10 <= tokens <= 30


def test_estimate_tokens_empty_string() -> None:
    cm = ContextManager()
    assert cm.estimate_tokens("") == 0


# ── build_context_window ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_context_window_structure() -> None:
    """build_context_window debe anteponer resúmenes y luego obs activas."""
    cm = ContextManager(max_recent_observations=2)

    async def fake_summarize(obs: Observation) -> str:
        return f"Resumen paso {obs.step_id}"

    cm._summarize = fake_summarize  # type: ignore[method-assign]

    for i in range(1, 4):  # 3 obs, max=2 → 1 resumida, 2 activas
        await cm.add_observation(_make_obs(i))

    messages = cm.build_context_window()

    assert len(messages) == 3  # 1 summary block + 2 active obs
    assert messages[0]["role"] == "user"
    assert "Resumen de observaciones anteriores" in messages[0]["content"]
    assert "Observación del paso 2" in messages[1]["content"]
    assert "Observación del paso 3" in messages[2]["content"]


@pytest.mark.asyncio
async def test_build_context_window_no_summaries() -> None:
    """Sin resúmenes, build_context_window solo retorna las obs activas."""
    cm = ContextManager(max_recent_observations=5)
    for i in range(1, 4):
        await cm.add_observation(_make_obs(i))

    messages = cm.build_context_window()
    assert len(messages) == 3
    assert all("Observación del paso" in m["content"] for m in messages)


# ── get_budget_report ─────────────────────────────────────────────────────────


def test_get_budget_report_empty() -> None:
    cm = ContextManager()
    report = cm.get_budget_report()
    assert report == {
        "total_observations": 0,
        "summarized_count": 0,
        "active_count": 0,
        "estimated_total_tokens": 0,
    }
