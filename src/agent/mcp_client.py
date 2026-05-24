"""MCP client hub — gestiona múltiples conexiones stdio para el orchestrator."""
from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import AnyUrl

logger = logging.getLogger(__name__)


class MCPHub:
    """Gestiona conexiones a múltiples MCP servers stdio.

    Indexa las tools de cada server al conectarse para enrutar llamadas sin
    ambigüedad y sin consultar el server en cada invocación.
    Compatible con ``async with`` para gestión automática del ciclo de vida.

    Ejemplo::

        async with MCPHub() as hub:
            await hub.connect("nexaplay_api", sys.executable,
                              ["-m", "src.skills.nexaplay_api"])
            await hub.connect("codegen", sys.executable,
                              ["-m", "src.skills.codegen"])
            tools = await hub.list_all_tools()
            result = await hub.call("nexaplay_api_call", {...})
    """

    def __init__(self) -> None:
        self.sessions: dict[str, ClientSession] = {}
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._tool_to_session: dict[str, ClientSession] = {}

    async def connect(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Lanza un MCP server como subprocess stdio y registra sus tools.

        Args:
            name: Nombre lógico del server (clave para :meth:`read_resource`).
            command: Ejecutable a lanzar (ej. ``sys.executable``).
            args: Argumentos del comando (ej. ``["-m", "src.skills.nexaplay_api"]``).
            env: Entorno completo para el subprocess. ``None`` hereda el entorno actual.
            cwd: Directorio de trabajo del subprocess. ``None`` hereda el del proceso actual.
        """
        params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=cwd,
        )
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        session: ClientSession = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()

        self.sessions[name] = session

        result = await session.list_tools()
        for tool in result.tools:
            self._tool_to_session[tool.name] = session

        tool_names = [t.name for t in result.tools]
        logger.info("Conectado a server %r | tools indexadas: %s", name, tool_names)

    async def list_all_tools(self) -> list[dict[str, Any]]:
        """Devuelve todas las tools en formato Anthropic.

        Returns:
            Lista de dicts con las keys ``name``, ``description``, ``input_schema``,
            listos para pasarse directamente como ``tools`` en la API de Anthropic.
        """
        tools: list[dict[str, Any]] = []
        for session in self.sessions.values():
            result = await session.list_tools()
            for tool in result.tools:
                tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema,
                    }
                )
        return tools

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoca una tool por nombre y retorna el texto concatenado del resultado.

        Args:
            tool_name: Nombre de la tool registrada en algún server conectado.
            arguments: Argumentos para la tool según su ``inputSchema``.

        Returns:
            Concatenación del ``.text`` de todos los ``TextContent`` del resultado.

        Raises:
            KeyError: Si ``tool_name`` no está indexada en ningún server conectado.
        """
        session = self._tool_to_session.get(tool_name)
        if session is None:
            available = sorted(self._tool_to_session)
            raise KeyError(
                f"Tool {tool_name!r} no encontrada. "
                f"Tools disponibles: {available}"
            )

        result = await session.call_tool(tool_name, arguments)
        return "".join(
            part.text
            for part in result.content
            if hasattr(part, "text") and part.text
        )

    async def read_resource(self, server_name: str, uri: str) -> str:
        """Lee un resource de un server específico.

        Útil para obtener el spec OpenAPI (``nexaplay://openapi-spec``)
        o la guía de estilo de codegen (``codegen://style-guide``).

        Args:
            server_name: Nombre lógico registrado vía :meth:`connect`.
            uri: URI del resource (ej. ``"nexaplay://openapi-spec"``).

        Returns:
            Contenido del resource como string.

        Raises:
            KeyError: Si ``server_name`` no está registrado.
        """
        session = self.sessions.get(server_name)
        if session is None:
            available = sorted(self.sessions)
            raise KeyError(
                f"Server {server_name!r} no encontrado. "
                f"Servers disponibles: {available}"
            )

        result = await session.read_resource(AnyUrl(uri))
        return "".join(
            part.text
            for part in result.contents
            if hasattr(part, "text") and part.text
        )

    async def aclose(self) -> None:
        """Cierra todas las conexiones MCP y los subprocesses asociados."""
        await self._exit_stack.aclose()

    async def __aenter__(self) -> "MCPHub":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
