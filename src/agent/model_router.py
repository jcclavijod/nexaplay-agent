"""Selección de modelo Claude según tipo de tarea (ADR sección 5)."""
from __future__ import annotations

import os
from enum import Enum


class TaskType(Enum):
    """Tipos de tarea reconocidos por el router de modelos."""

    PLANNING = "PLANNING"
    REASONING = "REASONING"
    CODEGEN = "CODEGEN"
    SUMMARIZATION = "SUMMARIZATION"


_DEFAULT_MODELS: dict[TaskType, str] = {
    TaskType.PLANNING: "claude-sonnet-4-5",
    TaskType.REASONING: "claude-sonnet-4-5",
    TaskType.CODEGEN: "claude-sonnet-4-5",
    TaskType.SUMMARIZATION: "claude-haiku-4-5-20251001",
}

_DEFAULT_MAX_TOKENS: dict[TaskType, int] = {
    TaskType.PLANNING: 2048,
    TaskType.REASONING: 1024,
    TaskType.CODEGEN: 4096,
    TaskType.SUMMARIZATION: 1024,
}

_DEFAULT_TEMPERATURE: dict[TaskType, float] = {
    TaskType.PLANNING: 0.0,
    TaskType.REASONING: 0.0,
    TaskType.CODEGEN: 0.0,
    TaskType.SUMMARIZATION: 0.3,
}

_ENV_KEY: dict[TaskType, str] = {
    TaskType.PLANNING: "MODEL_PLANNING",
    TaskType.REASONING: "MODEL_REASONING",
    TaskType.CODEGEN: "MODEL_CODEGEN",
    TaskType.SUMMARIZATION: "MODEL_SUMMARIZATION",
}


def get_model(task: TaskType) -> str:
    """Retorna el modelo Claude para el tipo de tarea dado.

    El env var correspondiente (MODEL_PLANNING, MODEL_REASONING, MODEL_CODEGEN,
    MODEL_SUMMARIZATION) tiene precedencia sobre el default del ADR.

    Args:
        task: Tipo de tarea que determina el modelo a usar.

    Returns:
        Nombre del modelo Claude (ej. ``"claude-sonnet-4-5"``).
    """
    return os.getenv(_ENV_KEY[task], _DEFAULT_MODELS[task])


def get_max_tokens(task: TaskType) -> int:
    """Retorna el límite de tokens de salida para el tipo de tarea dado.

    Args:
        task: Tipo de tarea a consultar.

    Returns:
        Máximo de tokens de salida configurado para la tarea.
    """
    return _DEFAULT_MAX_TOKENS[task]


def get_temperature(task: TaskType) -> float:
    """Retorna la temperatura de sampling para el tipo de tarea dado.

    Args:
        task: Tipo de tarea a consultar.

    Returns:
        Temperatura de sampling entre 0.0 (determinista) y 1.0.
    """
    return _DEFAULT_TEMPERATURE[task]
