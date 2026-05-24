"""Tests for src/agent/planner.py."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.planner import (
    Plan,
    create_plan,
    evaluate_precondition,
    resolve_inputs,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

VALID_PLAN_DICT = {
    "goal": "Actualizar operational_limit de servicio 42",
    "steps": [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Leer configuración actual",
            "inputs": {
                "endpoint": "/services/42/config",
                "method": "GET",
                "params": {"client_id": "MX-01", "country": "MX"},
            },
            "precondition": None,
        }
    ],
    "success_criterion": "operational_limit >= min_allowed",
}


def _mock_response(text: str) -> MagicMock:
    content = MagicMock()
    content.text = text
    response = MagicMock()
    response.content = [content]
    return response


def _make_client(*texts: str) -> AsyncMock:
    client = MagicMock()
    client.messages.create = AsyncMock(
        side_effect=[_mock_response(t) for t in texts]
    )
    return client


# ---------------------------------------------------------------------------
# evaluate_precondition
# ---------------------------------------------------------------------------


def test_evaluate_precondition_none():
    assert evaluate_precondition(None, {}) is True


def test_evaluate_precondition_lt():
    # Demo case: operational_limit=92 < min_allowed=95 → True
    step_results = {1: {"data": {"operational_limit": 92, "min_allowed": 95}}}
    assert (
        evaluate_precondition(
            "$step1.data.operational_limit < $step1.data.min_allowed",
            step_results,
        )
        is True
    )


def test_evaluate_precondition_lt_false():
    step_results = {1: {"data": {"operational_limit": 97, "min_allowed": 95}}}
    assert (
        evaluate_precondition(
            "$step1.data.operational_limit < $step1.data.min_allowed",
            step_results,
        )
        is False
    )


def test_evaluate_precondition_gte():
    step_results = {1: {"data": {"x": 100, "y": 100}}}
    assert evaluate_precondition("$step1.data.x >= $step1.data.y", step_results) is True


def test_evaluate_precondition_eq():
    step_results = {1: {"data": {"version": 3}}}
    assert evaluate_precondition("$step1.data.version == 3", step_results) is True


def test_evaluate_precondition_ne():
    step_results = {1: {"data": {"status": "ok"}}}
    assert evaluate_precondition("$step1.data.status != 'error'", step_results) is True


def test_evaluate_precondition_missing_step_raises():
    with pytest.raises(ValueError, match="step 2"):
        evaluate_precondition("$step2.data.x < 10", {})


def test_evaluate_precondition_missing_path_raises():
    step_results = {1: {"data": {"x": 5}}}
    with pytest.raises(ValueError):
        evaluate_precondition("$step1.data.nonexistent < 10", step_results)


def test_evaluate_precondition_no_operator_raises():
    with pytest.raises(ValueError, match="operator"):
        evaluate_precondition("$step1.data.x", {1: {"data": {"x": 5}}})


# ---------------------------------------------------------------------------
# resolve_inputs
# ---------------------------------------------------------------------------


def test_resolve_inputs_replaces_refs():
    step_results = {1: {"data": {"value": 42}}}
    inputs = {"limit": "$step1.data.value", "static": "hello"}
    resolved = resolve_inputs(inputs, step_results)
    assert resolved == {"limit": 42, "static": "hello"}


def test_resolve_inputs_nested_dict():
    step_results = {1: {"data": {"standard_value": 96}}}
    inputs = {"body": {"operational_limit": "$step1.data.standard_value"}}
    resolved = resolve_inputs(inputs, step_results)
    assert resolved == {"body": {"operational_limit": 96}}


def test_resolve_inputs_whole_dict_ref():
    step_results = {1: {"data": {"a": 1, "b": 2}}}
    inputs = {"context": "$step1.data"}
    resolved = resolve_inputs(inputs, step_results)
    assert resolved == {"context": {"a": 1, "b": 2}}


def test_resolve_inputs_missing_ref_raises():
    with pytest.raises(ValueError):
        resolve_inputs({"x": "$step9.data.y"}, {})


# ---------------------------------------------------------------------------
# create_plan
# ---------------------------------------------------------------------------


async def test_create_plan_parses_valid_json():
    client = _make_client(json.dumps(VALID_PLAN_DICT))
    plan = await create_plan("Actualiza servicio 42", [], client)

    assert isinstance(plan, Plan)
    assert plan.goal == VALID_PLAN_DICT["goal"]
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "nexaplay_api_call"
    assert client.messages.create.call_count == 1


async def test_create_plan_retries_on_invalid_json():
    client = _make_client("esto no es json {{{", json.dumps(VALID_PLAN_DICT))
    plan = await create_plan("Actualiza servicio 42", [], client)

    assert isinstance(plan, Plan)
    assert client.messages.create.call_count == 2

    # Second call must echo the error back to the model
    second_call_messages = client.messages.create.call_args_list[1][1]["messages"]
    user_turns = [m for m in second_call_messages if m["role"] == "user"]
    assert any("JSON inválido" in m["content"] for m in user_turns)


async def test_create_plan_raises_after_two_failures():
    client = _make_client("bad json", "also bad json")
    with pytest.raises(ValueError, match="PLAN_GENERATION_FAILED"):
        await create_plan("anything", [], client)

    assert client.messages.create.call_count == 2


async def test_create_plan_retries_on_invalid_schema():
    # Valid JSON but missing required fields → Pydantic raises → retry
    bad_schema = json.dumps({"goal": "ok"})  # missing steps, success_criterion
    client = _make_client(bad_schema, json.dumps(VALID_PLAN_DICT))
    plan = await create_plan("Actualiza servicio 42", [], client)

    assert isinstance(plan, Plan)
    assert client.messages.create.call_count == 2


async def test_create_plan_injects_tools_and_requirement(tmp_path, monkeypatch):
    """Verify the prompt template substitution reaches the model call."""
    client = _make_client(json.dumps(VALID_PLAN_DICT))
    tools = [{"name": "nexaplay_api_call", "description": "Calls the API"}]

    import src.agent.planner as planner_mod

    captured: list[dict] = []

    async def fake_create(**kwargs: object) -> MagicMock:
        captured.append(kwargs)
        return _mock_response(json.dumps(VALID_PLAN_DICT))

    client.messages.create = fake_create

    await create_plan("Fix service 42", tools, client)

    assert captured
    user_content = captured[0]["messages"][0]["content"]
    assert "nexaplay_api_call" in user_content
    assert "Fix service 42" in user_content
