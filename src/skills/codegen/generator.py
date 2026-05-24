"""Generador de código usando Claude API con validación y reintento."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Literal

import anthropic

# Fallback inline si src/agent/prompts/codegen.md aún no existe.
CODEGEN_PROMPT = """\
Eres un experto en desarrollo de software. Genera código de alta calidad para el \
siguiente requerimiento.

## Contexto técnico
```json
{technical_context}
```

## Requerimiento
{requirement}

## Lenguaje objetivo
{language}

Responde ÚNICAMENTE con un objeto JSON válido con estas keys exactas:
- "filename": nombre de archivo sugerido (ej: "service_validator.py")
- "code": el código completo, listo para ejecutar, sin bloques markdown
- "test": suite de tests unitarios para el código generado, sin bloques markdown
- "summary": descripción breve en español de qué hace el código (1-2 oraciones)

No incluyas backticks, markdown fences ni texto fuera del JSON. \
El JSON debe ser parseable directamente con json.loads().
"""

_PROMPT_PATH = Path(__file__).parents[2] / "agent" / "prompts" / "codegen.md"
_REQUIRED_KEYS = {"filename", "code", "test", "summary"}
_MODEL = "claude-sonnet-4-5"
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _load_prompt_template() -> str:
    """Carga el template desde disco o usa el fallback inline."""
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return CODEGEN_PROMPT


def _render_prompt(
    template: str,
    requirement: str,
    technical_context: dict,
    language: str,
) -> str:
    """Sustituye los placeholders en el template con los valores reales.

    Usa ``.replace()`` en lugar de ``.format()`` para evitar conflictos con
    llaves dentro del JSON serializado de ``technical_context``.

    Args:
        template: Template de prompt con los placeholders ``{technical_context}``,
            ``{requirement}`` y ``{language}``.
        requirement: Descripción del requerimiento de código.
        technical_context: Contexto técnico a serializar como JSON indentado.
        language: Lenguaje objetivo (``"python"`` o ``"typescript"``).

    Returns:
        Prompt listo para enviar a Claude.
    """
    return (
        template
        .replace("{technical_context}", json.dumps(technical_context, indent=2))
        .replace("{requirement}", requirement)
        .replace("{language}", language)
    )


def _clean_response(text: str) -> str:
    """Elimina markdown fences si Claude las incluyó en la respuesta.

    Args:
        text: Texto crudo de la respuesta de Claude.

    Returns:
        Texto sin fences, listo para parsear como JSON.
    """
    stripped = text.strip()
    m = _FENCE_RE.match(stripped)
    return m.group(1).strip() if m else stripped


def _parse_response(raw: str) -> dict:
    """Parsea el JSON de respuesta y valida que contenga las keys requeridas.

    Args:
        raw: Texto crudo de la respuesta (puede incluir markdown fences).

    Returns:
        Dict con al menos las keys ``filename``, ``code``, ``test`` y ``summary``.

    Raises:
        json.JSONDecodeError: Si el contenido no es JSON válido.
        ValueError: Si faltan keys obligatorias en el resultado.
    """
    result = json.loads(_clean_response(raw))
    missing = _REQUIRED_KEYS - result.keys()
    if missing:
        raise ValueError(f"Faltan keys en la respuesta de Claude: {missing}")
    return result


def _check_python_syntax(code: str) -> str | None:
    """Retorna el mensaje de error de ast.parse, o None si la sintaxis es válida.

    Args:
        code: Código Python a validar.

    Returns:
        Mensaje de error de ``SyntaxError``, o ``None`` si el código es válido.
    """
    try:
        ast.parse(code)
        return None
    except SyntaxError as exc:
        return str(exc)


async def _call_claude(client: anthropic.AsyncAnthropic, prompt: str) -> str:
    """Ejecuta una llamada a la API de Claude y retorna el texto de la respuesta.

    Args:
        client: Cliente ``anthropic.AsyncAnthropic`` configurado.
        prompt: Prompt completo a enviar al modelo.

    Returns:
        Texto de la respuesta del modelo (primer bloque de contenido).
    """
    response = await client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def generate(
    requirement: str,
    technical_context: dict,
    language: Literal["python", "typescript"] = "python",
) -> dict:
    """Genera código y tests usando Claude API.

    Carga el prompt desde src/agent/prompts/codegen.md (o usa el fallback inline),
    llama a Claude con temperature=0 y parsea la respuesta como JSON estructurado.
    Para código Python, valida la sintaxis con ast.parse y reintenta una vez si falla.

    Args:
        requirement: Descripción del requerimiento de código a generar.
        technical_context: Contexto técnico (schema, ejemplos, constraints).
            No puede estar vacío.
        language: Lenguaje objetivo ("python" o "typescript").

    Returns:
        Dict con keys: filename, code, test, summary.
        Si el código Python sigue siendo inválido tras el reintento, el campo
        "summary" lleva el prefijo "[UNVALIDATED] " (honesto según ADR sección 6).

    Raises:
        ValueError: Si technical_context está vacío.
        RuntimeError: Si Claude no produce JSON parseable con las keys requeridas.
    """
    if not technical_context:
        raise ValueError("INVALID_CONTEXT")

    template = _load_prompt_template()
    base_prompt = _render_prompt(template, requirement, technical_context, language)
    client = anthropic.AsyncAnthropic()

    # Primer intento: generación base.
    raw = await _call_claude(client, base_prompt)
    try:
        result = _parse_response(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("GENERATION_FAILED") from exc

    # Validación de sintaxis Python; TypeScript no se puede validar sin parser externo.
    if language != "python":
        return result

    parse_error = _check_python_syntax(result["code"])
    if parse_error is None:
        return result

    # Reintento con contexto de error de sintaxis.
    retry_prompt = (
        base_prompt
        + f"\n\nEl intento anterior produjo código no parseable: {parse_error}. Corrige."
    )
    raw2 = await _call_claude(client, retry_prompt)
    try:
        result = _parse_response(raw2)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("GENERATION_FAILED") from exc

    if _check_python_syntax(result["code"]) is not None:
        # Honesto es mejor que silencioso (ADR sección 6).
        result["summary"] = "[UNVALIDATED] " + result["summary"]

    return result
