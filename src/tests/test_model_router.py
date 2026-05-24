"""Tests para model_router."""
from __future__ import annotations

import pytest

from src.agent.model_router import TaskType, get_max_tokens, get_model, get_temperature


# ── Defaults ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "task, expected",
    [
        (TaskType.PLANNING, "claude-sonnet-4-5"),
        (TaskType.REASONING, "claude-sonnet-4-5"),
        (TaskType.CODEGEN, "claude-sonnet-4-5"),
        (TaskType.SUMMARIZATION, "claude-haiku-4-5-20251001"),
    ],
)
def test_get_model_defaults(task: TaskType, expected: str) -> None:
    assert get_model(task) == expected


@pytest.mark.parametrize(
    "task, expected",
    [
        (TaskType.PLANNING, 2048),
        (TaskType.REASONING, 1024),
        (TaskType.CODEGEN, 4096),
        (TaskType.SUMMARIZATION, 1024),
    ],
)
def test_get_max_tokens_defaults(task: TaskType, expected: int) -> None:
    assert get_max_tokens(task) == expected


@pytest.mark.parametrize(
    "task, expected",
    [
        (TaskType.PLANNING, 0.0),
        (TaskType.REASONING, 0.0),
        (TaskType.CODEGEN, 0.0),
        (TaskType.SUMMARIZATION, 0.3),
    ],
)
def test_get_temperature_defaults(task: TaskType, expected: float) -> None:
    assert get_temperature(task) == expected


# ── Env var overrides ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "task, env_key",
    [
        (TaskType.PLANNING, "MODEL_PLANNING"),
        (TaskType.REASONING, "MODEL_REASONING"),
        (TaskType.CODEGEN, "MODEL_CODEGEN"),
        (TaskType.SUMMARIZATION, "MODEL_SUMMARIZATION"),
    ],
)
def test_get_model_env_override(
    task: TaskType, env_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env_key, "claude-opus-4-7")
    assert get_model(task) == "claude-opus-4-7"
