"""Gestor de contexto de la ventana de observaciones (ADR sección 5)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import anthropic
import tiktoken
from pydantic import BaseModel, Field

from src.agent.model_router import TaskType, get_max_tokens, get_model, get_temperature

_ENCODING = tiktoken.get_encoding("cl100k_base")

_SUMMARIZATION_PROMPT = (
    "Resume esta observación en máximo 200 tokens preservando: "
    "tool usado, inputs principales, outcome (success/error), valores clave "
    "del resultado. Output en español, una sola frase."
)


class Observation(BaseModel):
    """Registro de una observación de un paso del plan.

    Almacena la tool invocada, sus argumentos, el resultado obtenido
    y metadatos de auditoría como timestamp y estimación de tokens.
    """

    step_id: int
    tool_name: str
    arguments: dict
    result: dict
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tokens_estimated: int = 0


class ContextManager:
    """Gestiona la ventana de contexto de observaciones para el agente ReAct.

    Mantiene las ``max_recent_observations`` más recientes en memoria activa.
    Las observaciones más antiguas se comprimen con Claude (resumen en ≤ 200 tokens)
    y se acumulan como texto en ``summarized_observations``.

    Ejemplo::

        ctx = ContextManager(max_recent_observations=3)
        await ctx.add_observation(obs)
        messages = ctx.build_context_window()
    """

    def __init__(
        self,
        max_recent_observations: int = 3,
        summarization_threshold_tokens: int = 2000,
    ) -> None:
        """Inicializa el gestor de contexto.

        Args:
            max_recent_observations: Número máximo de observaciones activas antes
                de comprimir las más antiguas.
            summarization_threshold_tokens: Umbral de tokens reservado para uso futuro.
        """
        self.max_recent_observations = max_recent_observations
        self.summarization_threshold_tokens = summarization_threshold_tokens
        self.observations: list[Observation] = []
        self.summarized_observations: list[str] = []
        self._client = anthropic.AsyncAnthropic()

    async def add_observation(self, obs: Observation) -> None:
        """Añade una observación y, si se supera el límite, comprime la más antigua.

        Args:
            obs: Observación del paso a registrar.
        """
        self.observations.append(obs)
        if len(self.observations) > self.max_recent_observations:
            oldest = self.observations.pop(0)
            summary = await self._summarize(oldest)
            self.summarized_observations.append(summary)

    async def _summarize(self, obs: Observation) -> str:
        """Comprime una observación a ≤ 200 tokens mediante una llamada a Claude.

        Args:
            obs: Observación a comprimir.

        Returns:
            Resumen en español de una sola frase con los datos esenciales.
        """
        obs_text = json.dumps(
            {
                "step_id": obs.step_id,
                "tool_name": obs.tool_name,
                "arguments": obs.arguments,
                "result": obs.result,
            },
            ensure_ascii=False,
        )
        model = get_model(TaskType.SUMMARIZATION)
        response = await self._client.messages.create(
            model=model,
            max_tokens=get_max_tokens(TaskType.SUMMARIZATION),
            temperature=get_temperature(TaskType.SUMMARIZATION),
            messages=[
                {
                    "role": "user",
                    "content": f"{_SUMMARIZATION_PROMPT}\n\nObservación:\n{obs_text}",
                }
            ],
        )
        return response.content[0].text.strip()

    def estimate_tokens(self, text: str) -> int:
        """Estima el número de tokens de *text* usando el encoding cl100k_base.

        Args:
            text: Texto a medir.

        Returns:
            Número estimado de tokens.
        """
        return len(_ENCODING.encode(text))

    def build_context_window(self) -> list[dict]:
        """Construye la lista de mensajes para incluir en el prompt del agente.

        Antepone un bloque de resumen si hay observaciones comprimidas, seguido
        de las observaciones activas en orden cronológico.

        Returns:
            Lista de dicts ``{"role": "user", "content": "..."}`` listos para
            pasarse a la API de Anthropic.
        """
        messages: list[dict] = []

        if self.summarized_observations:
            summary_block = "Resumen de observaciones anteriores:\n" + "\n".join(
                f"- {s}" for s in self.summarized_observations
            )
            messages.append({"role": "user", "content": summary_block})

        for obs in self.observations:
            result_json = json.dumps(obs.result, ensure_ascii=False)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Observación del paso {obs.step_id} ({obs.tool_name}): {result_json}"
                    ),
                }
            )

        return messages

    def get_budget_report(self) -> dict:
        """Retorna un informe del uso de tokens de la ventana de contexto.

        Returns:
            Dict con las keys ``total_observations``, ``summarized_count``,
            ``active_count`` y ``estimated_total_tokens``.
        """
        active_tokens = sum(
            self.estimate_tokens(
                json.dumps(
                    {
                        "tool_name": o.tool_name,
                        "arguments": o.arguments,
                        "result": o.result,
                    }
                )
            )
            for o in self.observations
        )
        summarized_tokens = sum(
            self.estimate_tokens(s) for s in self.summarized_observations
        )
        return {
            "total_observations": len(self.observations) + len(self.summarized_observations),
            "summarized_count": len(self.summarized_observations),
            "active_count": len(self.observations),
            "estimated_total_tokens": active_tokens + summarized_tokens,
        }
