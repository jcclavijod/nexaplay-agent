"""MCP server stdio — expone nexaplay_api_call y el OpenAPI spec de NexaPlay."""
from __future__ import annotations

import json
import os
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from .http_client import NexaPlayHTTPClient

_SPEC_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "specs"
    / "nexaplay-openapi.yaml"
)

server = Server("nexaplay-api")
_http_client: NexaPlayHTTPClient | None = None


def _get_client() -> NexaPlayHTTPClient:
    """Retorna la instancia singleton de :class:`NexaPlayHTTPClient`, creándola si es necesario."""
    global _http_client
    if _http_client is None:
        _http_client = NexaPlayHTTPClient(base_url=os.environ["NEXAPLAY_BASE_URL"])
    return _http_client


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Registra las tools expuestas por este server MCP.

    Returns:
        Lista con la tool ``nexaplay_api_call`` y su schema de inputs.
    """
    return [
        types.Tool(
            name="nexaplay_api_call",
            description=(
                "Ejecuta una llamada HTTP contra la API de NexaPlay con retry automático, "
                "idempotency-key determinística y clasificación estructurada de errores. "
                "GET es seguro reintentar libremente. "
                "POST incluye Idempotency-Key calculada desde job_id y step, garantizando "
                "exactamente una escritura incluso ante reintentos. "
                "Errores 4xx (VALIDATION_ERROR) no se reintentan. "
                "Errores 5xx (SERVER_ERROR) y de red/timeout (NETWORK_ERROR, TIMEOUT_ERROR) "
                "se reintentan hasta 3 veces con backoff progresivo de 1→2→4 segundos. "
                "SILENT_WRITE_FAILURE (2xx con datos no aplicados) se escala de inmediato."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "Path relativo del endpoint, ej: /services/42/config",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST"],
                        "description": "Método HTTP",
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            'Query params opcionales, ej: {"client_id": "MX-01", "country": "MX"}'
                        ),
                    },
                    "body": {
                        "type": "object",
                        "description": "Body JSON para POST. Ignorado en GET.",
                    },
                    "job_id": {
                        "type": "string",
                        "description": "ID del job activo, usado para calcular la Idempotency-Key.",
                    },
                    "step": {
                        "type": "integer",
                        "description": "Número de paso dentro del job, parte de la Idempotency-Key.",
                    },
                },
                "required": ["endpoint", "method", "job_id", "step"],
            },
        )
    ]


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """Registra los resources expuestos por este server MCP.

    Returns:
        Lista con el resource ``nexaplay://openapi-spec``.
    """
    return [
        types.Resource(
            uri=types.AnyUrl("nexaplay://openapi-spec"),  # type: ignore[arg-type]
            name="OpenAPI Spec de NexaPlay",
            description="Contrato completo de los endpoints disponibles",
            mimeType="application/yaml",
        )
    ]


@server.read_resource()
async def read_resource(uri: types.AnyUrl) -> str:
    """Lee el contenido de un resource por URI.

    Args:
        uri: URI del resource a leer. Solo se acepta ``"nexaplay://openapi-spec"``.

    Returns:
        Contenido del archivo YAML del spec OpenAPI de NexaPlay.

    Raises:
        ValueError: Si la URI no corresponde a ningún resource registrado.
    """
    if str(uri) != "nexaplay://openapi-spec":
        raise ValueError(f"Resource desconocido: {uri}")
    return _SPEC_PATH.read_text(encoding="utf-8")


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Despacha la invocación de una tool y retorna el resultado como JSON.

    Args:
        name: Nombre de la tool a invocar. Solo se acepta ``"nexaplay_api_call"``.
        arguments: Argumentos de la tool según su ``inputSchema``.

    Returns:
        Lista con un único :class:`~mcp.types.TextContent` cuyo texto es el JSON
        del resultado de :meth:`~http_client.NexaPlayHTTPClient.call`.

    Raises:
        ValueError: Si ``name`` no corresponde a ninguna tool registrada.
    """
    if name != "nexaplay_api_call":
        raise ValueError(f"Tool desconocida: {name}")

    result = await _get_client().call(
        endpoint=arguments["endpoint"],
        method=arguments["method"],
        job_id=arguments["job_id"],
        step=arguments["step"],
        params=arguments.get("params"),
        body=arguments.get("body"),
    )

    return [
        types.TextContent(
            type="text",
            text=json.dumps(result, ensure_ascii=False),
        )
    ]


async def main() -> None:
    """Punto de entrada del server MCP en modo stdio.

    Carga las variables de entorno con ``dotenv`` e inicia el loop de
    lectura/escritura stdio hasta que el cliente cierre la conexión.
    """
    from dotenv import load_dotenv

    load_dotenv()

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="nexaplay-api",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
