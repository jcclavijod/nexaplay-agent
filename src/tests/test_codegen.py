"""Tests para src/skills/codegen/generator.py."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CODE = "def hello():\n    return 'world'\n"

_VALID_PAYLOAD = {
    "filename": "hello.py",
    "code": _VALID_CODE,
    "test": "def test_hello():\n    assert hello() == 'world'\n",
    "summary": "Función simple que retorna world.",
}

# Código con error de sintaxis deliberado.
_INVALID_CODE_PAYLOAD = {
    "filename": "broken.py",
    "code": "def hello(:\n    return 'broken'\n",
    "test": "...",
    "summary": "Código con error de sintaxis.",
}


def _make_claude_response(payload: dict) -> MagicMock:
    """Construye un objeto que imita anthropic.types.Message con un text block."""
    block = MagicMock()
    block.text = json.dumps(payload)
    msg = MagicMock()
    msg.content = [block]
    return msg


@pytest.fixture
def mock_client(monkeypatch) -> MagicMock:
    """Parchea anthropic.AsyncAnthropic en el módulo generator y retorna el mock del cliente."""
    client = MagicMock()
    client.messages.create = AsyncMock()

    constructor = MagicMock(return_value=client)
    monkeypatch.setattr("src.skills.codegen.generator.anthropic.AsyncAnthropic", constructor)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_generate_retorna_cuatro_keys(mock_client):
    """generate() retorna un dict con exactamente las 4 keys requeridas."""
    mock_client.messages.create.return_value = _make_claude_response(_VALID_PAYLOAD)

    from src.skills.codegen.generator import generate

    result = await generate("Crea una función hello", {"schema": "simple"})

    assert {"filename", "code", "test", "summary"} <= result.keys()


async def test_generate_lanza_valor_error_con_contexto_vacio():
    """generate() lanza ValueError('INVALID_CONTEXT') si technical_context está vacío."""
    from src.skills.codegen.generator import generate

    with pytest.raises(ValueError, match="INVALID_CONTEXT"):
        await generate("algo", {})


async def test_generate_reintenta_cuando_codigo_no_parseable(mock_client):
    """Si el primer intento produce código inválido, se realiza un segundo llamado a Claude."""
    mock_client.messages.create.side_effect = [
        _make_claude_response(_INVALID_CODE_PAYLOAD),  # primer intento → sintaxis rota
        _make_claude_response(_VALID_PAYLOAD),           # reintento → código correcto
    ]

    from src.skills.codegen.generator import generate

    result = await generate("Crea algo", {"schema": "test"})

    assert mock_client.messages.create.call_count == 2
    assert "[UNVALIDATED]" not in result["summary"]


async def test_generate_unvalidated_cuando_ambos_intentos_fallan(mock_client):
    """Si ambos intentos producen código no parseable, retorna con '[UNVALIDATED] ' en summary."""
    mock_client.messages.create.return_value = _make_claude_response(_INVALID_CODE_PAYLOAD)

    from src.skills.codegen.generator import generate

    result = await generate("Crea algo", {"schema": "test"})

    assert mock_client.messages.create.call_count == 2
    assert result["summary"].startswith("[UNVALIDATED] ")


async def test_generate_runtime_error_cuando_json_invalido(mock_client):
    """Si Claude retorna texto no-JSON, generate() lanza RuntimeError('GENERATION_FAILED')."""
    block = MagicMock()
    block.text = "Lo siento, no puedo generar eso."
    bad_msg = MagicMock()
    bad_msg.content = [block]
    mock_client.messages.create.return_value = bad_msg

    from src.skills.codegen.generator import generate

    with pytest.raises(RuntimeError, match="GENERATION_FAILED"):
        await generate("Crea algo", {"schema": "test"})


async def test_generate_typescript_omite_validacion_ast(mock_client):
    """Para TypeScript no se ejecuta ast.parse y se devuelve el resultado sin reintento."""
    ts_payload = {
        "filename": "hello.ts",
        "code": "export function hello(): string { return 'world'; }",
        "test": "test('hello', () => { expect(hello()).toBe('world'); });",
        "summary": "Función hello en TypeScript.",
    }
    mock_client.messages.create.return_value = _make_claude_response(ts_payload)

    from src.skills.codegen.generator import generate

    result = await generate("Crea función hello", {"schema": "ts"}, language="typescript")

    assert mock_client.messages.create.call_count == 1
    assert result["filename"] == "hello.ts"
