"""Generación y evaluación de planes para el agente ReAct de NexaPlay."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.agent import model_router
from src.agent.model_router import TaskType

_PROMPT_PATH = Path(__file__).parent / "prompts" / "planner.md"

# Matches "$stepN.path.to.field" — anchored so partial tokens don't resolve.
_REF_RE = re.compile(r"^\$step(\d+)((?:\.\w+)+)$")

# Two-char operators must come before one-char to avoid premature matching.
_OP_RE = re.compile(r"\s*(<=|>=|!=|==|<|>)\s*")


class PlanStep(BaseModel):
    """Un paso atómico dentro de un :class:`Plan`.

    Describe qué tool invocar, con qué inputs y bajo qué precondición
    (expresión de comparación sobre resultados de pasos anteriores).
    """

    id: int
    tool: str
    purpose: str
    inputs: dict[str, Any]
    precondition: str | None = None


class Plan(BaseModel):
    """Plan de ejecución generado por Claude para satisfacer un requerimiento.

    Contiene la lista de :class:`PlanStep` a ejecutar en orden y el criterio
    que determina si el objetivo se alcanzó.
    """

    goal: str
    steps: list[PlanStep]
    success_criterion: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_ref(token: str, step_results: dict[int, dict]) -> Any:
    """Resuelve una referencia ``"$stepN.path"`` o parsea un literal escalar.

    Args:
        token: Cadena a resolver — puede ser una referencia ``$stepN.a.b``
            o un literal (int, float, o string sin comillas).
        step_results: Mapa de ``step_id → resultado`` de los pasos ya ejecutados.

    Returns:
        El valor resuelto (cualquier tipo JSON).

    Raises:
        ValueError: Si la referencia apunta a un paso sin resultado o si la
            ruta no existe en el resultado.
    """
    stripped = token.strip()
    m = _REF_RE.fullmatch(stripped)
    if m:
        step_id = int(m.group(1))
        path = m.group(2).lstrip(".")
        if step_id not in step_results:
            raise ValueError(
                f"Reference '{stripped}' points to step {step_id} which has no result yet"
            )
        value: Any = step_results[step_id]
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(
                    f"Path '.{path}' not found while resolving '{stripped}'"
                )
            value = value[part]
        return value

    # Scalar literal — try int, float, then bare string.
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped.strip('"').strip("'")


def _resolve_value(value: Any, step_results: dict[int, dict]) -> Any:
    """Resuelve recursivamente referencias ``$stepN.path`` en cualquier valor.

    Args:
        value: Valor a resolver — puede ser string, dict, list o escalar.
        step_results: Mapa de ``step_id → resultado`` de los pasos ya ejecutados.

    Returns:
        El valor con todas las referencias sustituidas por sus valores reales.
    """
    if isinstance(value, str) and _REF_RE.fullmatch(value.strip()):
        return _resolve_ref(value, step_results)
    if isinstance(value, dict):
        return {k: _resolve_value(v, step_results) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, step_results) for item in value]
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_precondition(
    precondition: str | None, step_results: dict[int, dict]
) -> bool:
    """Evalúa si *precondition* se cumple contra *step_results*.

    Soporta los operadores: ``< <= > >= == !=``.
    Las referencias usan la forma ``"$stepN.path.to.field"``.
    Pasar ``None`` equivale a "ejecutar siempre este paso".

    Args:
        precondition: Expresión de comparación o ``None`` para ejecución incondicional.
        step_results: Mapa de ``step_id → resultado`` de los pasos ya ejecutados.

    Returns:
        ``True`` si la precondición se cumple (o es ``None``), ``False`` en caso contrario.

    Raises:
        ValueError: Si la expresión no contiene ningún operador de comparación.
    """
    if precondition is None:
        return True

    m = _OP_RE.search(precondition)
    if not m:
        raise ValueError(
            f"No comparison operator found in precondition: {precondition!r}"
        )

    lhs = _resolve_ref(precondition[: m.start()], step_results)
    op = m.group(1)
    rhs = _resolve_ref(precondition[m.end() :], step_results)

    ops: dict[str, Any] = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    return ops[op](lhs, rhs)


def resolve_inputs(inputs: dict, step_results: dict[int, dict]) -> dict:
    """Retorna *inputs* con cada referencia ``"$stepN.path"`` sustituida por su valor.

    Args:
        inputs: Dict de inputs de un paso, posiblemente con referencias.
        step_results: Mapa de ``step_id → resultado`` de los pasos ya ejecutados.

    Returns:
        Copia de *inputs* con todas las referencias resueltas.
    """
    return {k: _resolve_value(v, step_results) for k, v in inputs.items()}


async def create_plan(
    requirement: str,
    available_tools: list[dict],
    anthropic_client: Any,
) -> Plan:
    """Genera un :class:`Plan` estructurado a partir de un requerimiento en lenguaje natural.

    Llama a Claude una vez; si hay fallo de JSON o validación, reintenta exactamente
    una vez echando el error de vuelta al modelo. Lanza
    ``ValueError("PLAN_GENERATION_FAILED")`` ante dos fallos consecutivos.

    Args:
        requirement: Descripción del objetivo en lenguaje natural.
        available_tools: Lista de tools en formato Anthropic (``name``, ``description``,
            ``input_schema``).
        anthropic_client: Cliente ``anthropic.AsyncAnthropic`` para llamar al modelo.

    Returns:
        Plan validado con los pasos a ejecutar.

    Raises:
        ValueError: Con el mensaje ``"PLAN_GENERATION_FAILED"`` si Claude genera JSON
            inválido en dos intentos consecutivos.
    """
    prompt = (
        _PROMPT_PATH.read_text(encoding="utf-8")
        .replace("{tools_json}", json.dumps(available_tools, indent=2))
        .replace("{requirement}", requirement)
    )

    model = model_router.get_model(TaskType.PLANNING)
    temperature = model_router.get_temperature(TaskType.PLANNING)
    max_tokens = model_router.get_max_tokens(TaskType.PLANNING)

    messages: list[dict] = [{"role": "user", "content": prompt}]
    raw = ""
    last_error = ""

    for attempt in range(2):
        if attempt == 1:
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"El intento anterior produjo JSON inválido: {last_error}. "
                            "Devuelve JSON puro."
                        ),
                    },
                ]
            )

        response = await anthropic_client.messages.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        raw = response.content[0].text.strip()

        try:
            data = json.loads(raw)
            return Plan(**data)
        except Exception as exc:
            last_error = str(exc)
            if attempt == 1:
                raise ValueError("PLAN_GENERATION_FAILED") from exc

    raise ValueError("PLAN_GENERATION_FAILED")  # unreachable; satisfies type checkers
