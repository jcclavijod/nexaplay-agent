"""Generador de código usando Claude API con validación y reintento."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Literal

import anthropic
import logging  # DEBUG_TEMP

logger = logging.getLogger(__name__)  # DEBUG_TEMP

# Fallback inline si src/agent/prompts/codegen.md aún no existe.
CODEGEN_PROMPT = """\
Eres un generador de módulos de código de producción. Recibes un requerimiento
funcional y un contexto técnico que describe el schema real de la API.


# REGLAS INNEGOCIABLES (HIGHEST PRIORITY)
- NO uses markdown
- NO uses ``` fences
- NO agregues texto antes o después del JSON
- NO expliques nada
- SOLO responde con JSON válido
- El JSON debe ser parseable directamente con json.loads()
- 
# Reglas de ingeniería
- Usa únicamente campos presentes en `technical_context`. Si un campo no está,
  no lo inventes.
- Genera código que pase mypy estricto en Python o tsc strict en TypeScript.
- Incluye type hints / tipos explícitos en toda firma pública.
- Incluye al menos un test unitario que valide el camino feliz y un camino
  de error.
- Comentarios en español, identificadores en inglés.
- Para Python: usa `httpx`, `pydantic`, `pytest`. No traigas dependencias nuevas.
- Para TypeScript: usa `fetch` nativo, sin axios.

# Formato de salida
Responde exactamente en este formato JSON, sin markdown, sin texto extra:

{
  "filename": "string — nombre sugerido del archivo, ej. service_config_updater.py",
  "code": "string — código completo del módulo, listo para ejecutarse",
  "test": "string — código completo del test unitario",
  "summary": "string — 2-3 frases en español describiendo qué hace el módulo"
}



# Contexto técnico (schema real)
{technical_context}

# Requerimiento
<user_requirement>{requirement}</user_requirement>

# Lenguaje objetivo
{language}
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
    text = text.strip()

    # Quitar fences si existen
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


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
    required = {"filename", "code", "test", "summary"}

    cleaned = _clean_response(raw)

    # 1. Intento directo (caso ideal)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # 2. Fallback: extraer JSON por balance de llaves
        start = cleaned.find("{")
        if start == -1:
            raise RuntimeError(f"GENERATION_FAILED: No JSON found\nRAW={raw[:800]}")

        brace = 0
        end = -1

        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                brace += 1
            elif cleaned[i] == "}":
                brace -= 1
                if brace == 0:
                    end = i
                    break

        if end == -1:
            raise RuntimeError(f"GENERATION_FAILED: Unbalanced JSON\nRAW={raw[:800]}")

        try:
            result = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"GENERATION_FAILED: Invalid JSON\nRAW={raw[:800]}") from exc

    # 3. Validación de keys
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Missing keys: {missing}")

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
    
    api_response = await client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    logger.debug("CLAUDE_RESPONSE_CONTENT=%r", api_response.content)

    parts = []

    for block in api_response.content:
        if hasattr(block, "text"):
            parts.append(block.text)

    return "\n".join(parts)


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
    logger.debug("CODEGEN_RAW_RESPONSE len=%d text=%r", len(raw), raw[:800])  # DEBUG_TEMP
    try:
        result = _parse_response(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.debug("CODEGEN_PARSE_ERROR_FIRST_ATTEMPT error=%r", exc)  # DEBUG_TEMP
        try:  # DEBUG_TEMP
            Path.home().joinpath("codegen_raw_fail.txt").write_text(  # DEBUG_TEMP
                f"ERROR: {exc}\n\nRAW ({len(raw)} chars):\n{raw}", encoding="utf-8"  # DEBUG_TEMP
            )  # DEBUG_TEMP
        except Exception:  # DEBUG_TEMP
            pass  # DEBUG_TEMP
        raise RuntimeError("GENERATION_FAILED") from exc

    # Validación de sintaxis Python; TypeScript no se puede validar sin parser externo.
    if language != "python":
        return result

    parse_error = _check_python_syntax(result["code"])
    if parse_error is None:
        return result

    # Reintento con contexto de error de sintaxis.
    logger.debug("CODEGEN_SYNTAX_RETRY_TRIGGERED parse_error=%r", parse_error)  # DEBUG_TEMP
    retry_prompt = (
        base_prompt
        + f"\n\nEl intento anterior produjo código no parseable: {parse_error}. Corrige."
    )
    raw2 = await _call_claude(client, retry_prompt)
    logger.debug("CODEGEN_RETRY_RAW_RESPONSE len=%d text=%r", len(raw2), raw2[:800])  # DEBUG_TEMP
    try:
        result = _parse_response(raw2)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("GENERATION_FAILED") from exc

    if _check_python_syntax(result["code"]) is not None:
        # Honesto es mejor que silencioso (ADR sección 6).
        result["summary"] = "[UNVALIDATED] " + result["summary"]

    return result
