"""MCP server stdio — expone code_generator y la guía de estilo como recurso."""
from __future__ import annotations

import json

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from .generator import generate

_STYLE_GUIDE = (
    "Comentarios en español, identificadores en inglés. "
    "Python: httpx + pydantic + pytest. TypeScript: fetch nativo. "
    "Type hints estrictos. Tests cubren happy path y error path."
)

server = Server("code-generator")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Registra las tools expuestas por este server MCP.

    Returns:
        Lista con la tool ``code_generator`` y su schema de inputs.
    """
    return [
        types.Tool(
            name="code_generator",
            description=(
                "Genera código funcional con test unitario usando como única fuente de verdad "
                "el technical_context. No inventa campos que no estén presentes en el contexto."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "requirement": {
                        "type": "string",
                        "description": "Descripción del requerimiento de código a generar.",
                    },
                    "technical_context": {
                        "type": "object",
                        "description": (
                            "Schema real del GET u otro contexto técnico estructurado. "
                            "El generador se limita a los campos presentes aquí."
                        ),
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "typescript"],
                        "description": "Lenguaje objetivo. Por defecto python.",
                    },
                },
                "required": ["requirement", "technical_context"],
            },
        )
    ]


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """Registra los resources expuestos por este server MCP.

    Returns:
        Lista con el resource ``codegen://style-guide``.
    """
    return [
        types.Resource(
            uri=types.AnyUrl("codegen://style-guide"),  # type: ignore[arg-type]
            name="Guía de estilo de código",
            description="Convenciones de idioma, librerías y cobertura de tests",
            mimeType="text/plain",
        )
    ]


@server.read_resource()
async def read_resource(uri: types.AnyUrl) -> str:
    """Lee el contenido de un resource por URI.

    Args:
        uri: URI del resource a leer. Solo se acepta ``"codegen://style-guide"``.

    Returns:
        Contenido del resource como string.

    Raises:
        ValueError: Si la URI no corresponde a ningún resource registrado.
    """
    if str(uri) != "codegen://style-guide":
        raise ValueError(f"Resource desconocido: {uri}")
    return _STYLE_GUIDE


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Despacha la invocación de una tool y retorna el resultado como JSON.

    Args:
        name: Nombre de la tool a invocar. Solo se acepta ``"code_generator"``.
        arguments: Argumentos de la tool según su ``inputSchema``.

    Returns:
        Lista con un único :class:`~mcp.types.TextContent` cuyo texto es un JSON
        con ``success=True`` y los campos del resultado, o ``success=False`` y
        ``error`` en caso de fallo.

    Raises:
        ValueError: Si ``name`` no corresponde a ninguna tool registrada.
    """
    if name != "code_generator":
        raise ValueError(f"Tool desconocida: {name}")

    try:
        result = await generate(
            requirement=arguments["requirement"],
            technical_context=arguments["technical_context"],
            language=arguments.get("language", "python"),
        )
        payload = {"success": True, **result}
    except KeyError as exc:
        payload = {"success": False, "error": f"MISSING_ARGUMENT: {exc}"}
    except (ValueError, RuntimeError) as exc:
        payload = {"success": False, "error": str(exc)}

    return [
        types.TextContent(
            type="text",
            text=json.dumps(payload, ensure_ascii=False),
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
                server_name="code-generator",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
