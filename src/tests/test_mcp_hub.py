"""Integration tests para MCPHub.

Arranca los dos MCP servers reales (nexaplay_api y codegen) como subprocesses
stdio y verifica que el hub conecta, indexa tools y enruta llamadas correctamente.

Estrategia para evitar calls reales:
- nexaplay_api_call: NEXAPLAY_BASE_URL apunta a un servidor HTTP local que devuelve
  404 inmediatamente → VALIDATION_ERROR (no retriable), respuesta rápida sin red.
- code_generator: ANTHROPIC_API_KEY puede ser inválida; el MCP SDK captura la
  AuthenticationError del servidor y la devuelve como CallToolResult(isError=True)
  con texto del error → MCPHub.call() retorna string no vacío.

Nota: en tests de scope function, usa ``monkeypatch.setenv`` para aislar variables
de entorno. Para fixtures de scope module se pasan directamente en el env del
subprocess vía ``MCPHub.connect(..., env={...})``.
"""
from __future__ import annotations

import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import pytest_asyncio

from src.agent.mcp_client import MCPHub

pytestmark = pytest.mark.asyncio(loop_scope="module")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Mock HTTP server ──────────────────────────────────────────────────────────


class _QuickRejectHandler(BaseHTTPRequestHandler):
    """Devuelve 404 a cualquier petición GET — fuerza VALIDATION_ERROR (no retry)."""

    def do_GET(self) -> None:
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'{"error":"not found"}')

    def log_message(self, *args: object) -> None:
        pass  # suprime output de http.server en la consola de tests


@pytest.fixture(scope="module")
def mock_nexaplay_url() -> str:
    """URL de un servidor HTTP local que responde 404 a todo."""
    httpd = HTTPServer(("127.0.0.1", 0), _QuickRejectHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


# ── MCPHub fixture ────────────────────────────────────────────────────────────


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def hub(mock_nexaplay_url: str) -> MCPHub:
    """MCPHub con nexaplay_api y codegen conectados y listos."""
    env = {
        **os.environ,
        "NEXAPLAY_BASE_URL": mock_nexaplay_url,
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "test-invalid"),
    }
    try:
        async with MCPHub() as h:
            await h.connect(
                name="nexaplay_api",
                command=sys.executable,
                args=["-m", "src.skills.nexaplay_api"],
                env=env,
                cwd=str(PROJECT_ROOT),
            )
            await h.connect(
                name="codegen",
                command=sys.executable,
                args=["-m", "src.skills.codegen"],
                env=env,
                cwd=str(PROJECT_ROOT),
            )
            yield h
    except RuntimeError as exc:
        # Workaround: anyio puede lanzar "Attempted to exit cancel scope" al
        # cerrar subprocesses en teardown; no indica fallo real del test.
        if "Attempted to exit cancel scope" not in str(exc):
            raise


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_connect_lists_tools_from_both_servers(hub: MCPHub) -> None:
    """list_all_tools devuelve las tools de nexaplay_api y codegen."""
    tools = await hub.list_all_tools()
    names = {t["name"] for t in tools}

    assert "nexaplay_api_call" in names
    assert "code_generator" in names


async def test_call_routes_to_correct_server(hub: MCPHub) -> None:
    """call() enruta cada tool al server correcto y devuelve respuesta no vacía.

    nexaplay_api_call: endpoint ``/nonexistent`` → HTTP 404 desde el mock server
    → VALIDATION_ERROR serializado en JSON → string no vacío.

    code_generator: API key puede ser inválida → AuthenticationError capturada
    por el MCP SDK → CallToolResult(isError=True) con texto del error → no vacío.
    """
    nexaplay_result = await hub.call(
        "nexaplay_api_call",
        {
            "endpoint": "/nonexistent",
            "method": "GET",
            "job_id": "test-routing-job",
            "step": 1,
        },
    )
    assert nexaplay_result, "nexaplay_api_call debe retornar texto no vacío"

    codegen_result = await hub.call(
        "code_generator",
        {
            "requirement": "función que suma dos enteros",
            "technical_context": {"inputs": ["a: int", "b: int"], "output": "int"},
        },
    )
    assert codegen_result, "code_generator debe retornar texto no vacío"
