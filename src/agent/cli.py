"""CLI entry-point for the NexaPlay AI agent."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

import anthropic
import tiktoken
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.tree import Tree

from src.agent.mcp_client import MCPHub
from src.agent.orchestrator import Orchestrator
from src.agent.planner import Plan, PlanStep

_ENCODING = tiktoken.get_encoding("cl100k_base")


class RichOrchestrator(Orchestrator):
    """Subclase de :class:`Orchestrator` que renderiza la ejecución con Rich.

    Sobreescribe todos los hooks de display para producir paneles, árboles
    y colores en la terminal. En modo ``dry_run`` los POSTs son simulados
    sin ejecutarse en el server.

    Ejemplo::

        orch = RichOrchestrator(hub, client, max_iterations=15, console=console)
        result = await orch.run("Listar servicios activos")
        orch.print_footer(result)
    """

    def __init__(
        self,
        hub: MCPHub,
        anthropic_client: Any,
        max_iterations: int,
        console: Console,
        dry_run: bool = False,
    ) -> None:
        """Inicializa el orchestrator Rich.

        Args:
            hub: Hub de MCP con los servers conectados.
            anthropic_client: Cliente ``anthropic.AsyncAnthropic`` para llamadas al modelo.
            max_iterations: Límite de pasos ejecutados antes de abortar el job.
            console: Consola Rich para el output de progreso.
            dry_run: Si es ``True``, los POSTs son simulados sin ejecutarse.
        """
        super().__init__(hub, anthropic_client, max_iterations, console=None)
        self._rich = console
        self._dry_run = dry_run
        self._start_ts: float = time.time()
        self._step_count: int = 0
        self._tool_call_count: int = 0

    # ------------------------------------------------------------------ display

    def _print(self, msg: Any) -> None:
        """Renderiza *msg* en la consola Rich.

        Si *msg* es un :class:`~rich.tree.Tree`, lo envuelve en un Panel de plan;
        si es string lo delega a :meth:`_render_string`; cualquier otro tipo se
        imprime directamente.
        """
        if isinstance(msg, Tree):
            self._rich.print(
                Panel(
                    msg,
                    title="[bold cyan]Plan de Ejecución[/]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
        elif isinstance(msg, str):
            self._render_string(msg)
        else:
            self._rich.print(msg)

    def _render_string(self, msg: str) -> None:
        """Aplica estilo Rich a *msg* según su contenido semántico.

        Detecta palabras clave como ``[Resumen]``, ``[FATAL]`` o ``"devolvió error"``
        para elegir color, icono y panel apropiados.
        """
        text = msg.strip()
        if not text:
            return
        if "[Resumen]" in text:
            summary = text.split("[Resumen]", 1)[1].strip()
            self._rich.print(
                Panel(
                    escape(summary),
                    title="[bold green]Resumen Final[/]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        elif "[FATAL]" in text:
            self._rich.print(
                Panel(escape(text), title="[bold red]Error Fatal[/]", border_style="red")
            )
        elif "devolvió error" in text:
            self._rich.print(f"[bold red]{escape(text)}[/]")
        elif "Omitido por precondición" in text:
            self._rich.print(f"[yellow]{escape(text)}[/]")
        elif "Artefactos escritos" in text:
            self._rich.print(f"[green]{escape(text)}[/]")
        else:
            self._rich.print(f"[dim]{escape(text)}[/]")

    # ------------------------------------------------------------------ plan

    def _format_plan_for_display(self, plan: Plan) -> Tree:
        """Retorna el plan formateado como árbol Rich para renderizado en panel.

        Args:
            plan: Plan a formatear.

        Returns:
            :class:`~rich.tree.Tree` con el objetivo, criterio de éxito y pasos.
        """
        tree = Tree(f"[bold white]{escape(plan.goal)}[/]")
        tree.add(f"[dim]Criterio de éxito: {escape(plan.success_criterion)}[/]")
        for s in plan.steps:
            precond = (
                f" [yellow dim]  [if: {escape(s.precondition)}][/]"
                if s.precondition
                else ""
            )
            node = tree.add(
                f"[dim white]Paso {s.id}[/] → [bold blue]{escape(s.tool)}[/]{precond}"
            )
            node.add(f"[italic dim]{escape(s.purpose)}[/]")
        return tree

    # ------------------------------------------------------------------ step hooks

    def _on_step_start(self, step: PlanStep, resolved: dict) -> None:
        """Muestra un Panel Rich con el Thought y Action del paso antes de ejecutarlo.

        Omite los inputs reservados (``job_id``, ``step``) y trunca los campos
        de payload grande (``body``, ``code``, ``test``, ``technical_context``).
        """
        self._step_count += 1
        self._tool_call_count += 1
        # Show all args except injected bookkeeping and large payloads.
        _skip = {"job_id", "step"}
        _truncate = {"technical_context", "body", "code", "test"}
        preview: dict = {}
        for k, v in resolved.items():
            if k in _skip:
                continue
            if k in _truncate:
                preview[k] = f"<{type(v).__name__}, omitted>"
            else:
                preview[k] = v
        content = (
            f"[italic dim]Thought:[/]  {escape(step.purpose)}\n"
            f"[blue]Action:[/]   [bold]{escape(step.tool)}[/]  "
            + escape(json.dumps(preview, ensure_ascii=False))
        )
        self._rich.print(
            Panel(
                content,
                title=f"[bold white]Paso {step.id}[/] — [blue]{escape(step.tool)}[/]",
                border_style="blue",
                subtitle="[dim]ejecutando…[/]",
            )
        )

    def _on_step_result(self, step: PlanStep, resolved: dict, result_data: dict) -> None:
        """Muestra un Panel Rich con el Observation del paso recién ejecutado.

        Renderiza en verde si fue exitoso, en rojo si falló, e incluye campos
        clave del resultado cuando están presentes en ``result_data["data"]``.
        """
        success = result_data.get("success", False)
        color = "green" if success else "red"
        icon = "✓" if success else "✗"
        lines = [f"[{color}]{icon}  success = {success}[/]"]

        if not success:
            err = result_data.get("error") or result_data.get("message")
            if err:
                lines.append(f"[red]  error : {escape(str(err))}[/]")
            else:
                # No error key — dump raw result so nothing is silently swallowed.
                raw = json.dumps(result_data, ensure_ascii=False)
                lines.append(f"[red]  raw   : {escape(raw[:300])}[/]")
        elif isinstance(result_data.get("data"), dict):
            data: dict = result_data["data"]
            for key in (
                "operational_limit",
                "min_allowed",
                "standard_value",
                "max_transactions_per_second",
                "version",
                "timestamp",
            ):
                if key in data:
                    v = data[key]
                    v_str = (
                        json.dumps(v, ensure_ascii=False)
                        if isinstance(v, (dict, list))
                        else str(v)
                    )
                    lines.append(f"  [dim]{key}:[/] [cyan]{escape(v_str)}[/]")
            if "updated" in data:
                lines.append(
                    "  [dim]updated:[/] "
                    + escape(json.dumps(data["updated"], ensure_ascii=False))
                )

        self._rich.print(
            Panel(
                "\n".join(lines),
                title=f"[{color}]Observation — Paso {step.id}[/]",
                border_style=color,
            )
        )

    # ------------------------------------------------------------------ POST confirmation

    def _request_post_confirmation(self, step_id: int, resolved: dict) -> bool:
        """Muestra el body del POST y solicita confirmación interactiva al usuario.

        En modo ``dry_run`` confirma automáticamente sin preguntar.
        Si la variable ``NEXAPLAY_AUTO_CONFIRM=1`` está activa, también confirma
        automáticamente (útil en CI/smoke tests).

        Args:
            step_id: Número del paso que origina el POST.
            resolved: Inputs resueltos del paso con ``endpoint`` y ``body``.

        Returns:
            ``True`` si el usuario (o el modo automático) confirma el POST.
        """
        endpoint = resolved.get("endpoint", "")
        body = resolved.get("body", {})
        body_json = json.dumps(body, indent=2, ensure_ascii=False)

        if self._dry_run:
            self._rich.print(
                Panel(
                    f"[yellow]DRY-RUN: POST simulado — no se ejecutará realmente[/]\n\n"
                    f"[dim]endpoint:[/] {escape(endpoint)}\n\n"
                    f"{escape(body_json)}",
                    title=f"[bold yellow]Paso {step_id} — POST (DRY-RUN)[/]",
                    border_style="yellow",
                )
            )
            return True

        self._rich.print(
            Panel(
                Syntax(body_json, "json", theme="monokai", word_wrap=True),
                title=f"[bold yellow]Paso {step_id} — Confirmar POST[/]",
                subtitle=f"[dim]endpoint: {escape(endpoint)}[/]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        # NEXAPLAY_AUTO_CONFIRM=1 skips the interactive prompt (useful for CI/smoke tests).
        if os.getenv("NEXAPLAY_AUTO_CONFIRM") == "1":
            self._rich.print("[yellow dim]AUTO_CONFIRM activo — POST confirmado automáticamente.[/]")
            return True

        answer = Prompt.ask(
            "\n[yellow bold]¿Confirmar esta escritura?[/] "
            "Escribe [bold]confirmar[/] para proceder",
            console=self._rich,
        )
        return answer.strip() == "confirmar"

    # ------------------------------------------------------------------ dry-run mock

    async def _call_tool(self, tool_name: str, arguments: dict) -> str:
        """Intercepta llamadas POST en modo dry_run y devuelve una respuesta simulada.

        En cualquier otro caso delega a la implementación base.

        Args:
            tool_name: Nombre de la tool a invocar.
            arguments: Argumentos resueltos del paso.

        Returns:
            JSON del resultado real o simulado como string.
        """
        method = str(arguments.get("method") or "").upper()
        if self._dry_run and tool_name == "nexaplay_api_call" and method == "POST":
            return json.dumps(
                {
                    "success": True,
                    "data": {
                        "service_id": arguments.get("service_id"),
                        "client_id": arguments.get("client_id"),
                        "previous": {},
                        "updated": arguments.get("body", {}),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "version": 1,
                    },
                    "_dry_run": True,
                }
            )
        return await super()._call_tool(tool_name, arguments)

    # ------------------------------------------------------------------ footer

    def print_footer(self, result: dict) -> None:
        """Imprime el panel de métricas de ejecución al finalizar el job.

        Muestra status, pasos ejecutados, número de tool calls, estimación
        de tokens y tiempo total.

        Args:
            result: Resultado del job retornado por :meth:`~Orchestrator.run`.
        """
        obs_text = json.dumps(result.get("observations", []), ensure_ascii=False)
        tokens_est = len(_ENCODING.encode(obs_text))
        elapsed = time.time() - self._start_ts
        status = result.get("status", "unknown")
        status_color = {"completed": "green", "aborted": "yellow", "error": "red"}.get(
            status, "white"
        )
        self._rich.print(
            Panel(
                f"[dim]Status:[/]         [{status_color}]{status}[/]\n"
                f"[dim]Pasos ejecutados:[/] {self._step_count}\n"
                f"[dim]Tool calls:[/]      {self._tool_call_count}\n"
                f"[dim]Tokens (est.):[/]   {tokens_est}\n"
                f"[dim]Tiempo total:[/]    [bold]{elapsed:.1f}s[/]",
                title="[dim]Métricas de Ejecución[/]",
                border_style="dim",
            )
        )


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------


async def _run(requirement: str, dry_run: bool, verbose: bool) -> int:  # noqa: ARG001
    """Configura el entorno, conecta los MCP servers y ejecuta el requerimiento.

    Args:
        requirement: Descripción del objetivo en lenguaje natural.
        dry_run: Si es ``True``, los POSTs son simulados.
        verbose: Si es ``True``, activa logs de debug (parámetro reservado, aún sin uso).

    Returns:
        Código de salida: ``0`` si el job completó, ``1`` si abortó o falló,
        ``130`` si fue interrumpido por el usuario.
    """
    console = Console(legacy_windows=False)

    console.print(
        Panel(
            f"[bold cyan]{escape(requirement)}[/]",
            title="[bold white]NexaPlay AI Agent[/]",
            subtitle="[dim]powered by Claude Sonnet[/]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    if dry_run:
        console.print("[yellow dim]Modo DRY-RUN activo — los POSTs no se ejecutarán.[/]\n")

    anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    max_iterations = int(os.getenv("MAX_ITERATIONS", "15"))

    try:
        async with MCPHub() as hub:
            await hub.connect(
                "nexaplay-api", sys.executable, ["-m", "src.skills.nexaplay_api"]
            )
            await hub.connect("codegen", sys.executable, ["-m", "src.skills.codegen"])

            orchestrator = RichOrchestrator(
                hub=hub,
                anthropic_client=anthropic_client,
                max_iterations=max_iterations,
                console=console,
                dry_run=dry_run,
            )
            result = await orchestrator.run(requirement)
            orchestrator.print_footer(result)

        return 0 if result.get("status") == "completed" else 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Ejecución interrumpida por el usuario (Ctrl+C).[/]")
        return 130


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Punto de entrada principal de la CLI.

    Parsea argumentos, valida variables de entorno requeridas y lanza
    el loop asyncio con :func:`_run`. Soporta modo interactivo cuando
    no se pasa ``requirement`` como argumento posicional.
    """
    # Force UTF-8 stdout so rich can render Unicode on Windows terminals.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="NexaPlay AI Agent — agente de desarrollo AI-Native",
        prog="python -m src.agent.cli",
    )
    parser.add_argument(
        "requirement",
        nargs="?",
        help="Requerimiento en lenguaje natural (omitir para modo interactivo)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simula POSTs sin ejecutarlos realmente",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activa logs de debug del MCP client y del agente",
    )
    args = parser.parse_args()

    log_level = (
        logging.DEBUG
        if args.verbose
        else logging.getLevelName(os.getenv("LOG_LEVEL", "WARNING"))
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    missing = [v for v in ("ANTHROPIC_API_KEY", "NEXAPLAY_BASE_URL") if not os.getenv(v)]
    if missing:
        err = Console(stderr=True)
        err.print(
            f"[bold red]Error:[/] Variables de entorno requeridas no configuradas: "
            f"[bold]{', '.join(missing)}[/]\n"
            "  Copia [bold].env.example[/] → [bold].env[/] y rellena los valores."
        )
        sys.exit(1)

    requirement = args.requirement
    if not requirement:
        prompt_console = Console()
        requirement = Prompt.ask(
            "[bold cyan]Ingresa el requerimiento[/]", console=prompt_console
        )
        if not requirement.strip():
            prompt_console.print("[red]El requerimiento no puede estar vacío.[/]")
            sys.exit(1)

    try:
        exit_code = asyncio.run(_run(requirement.strip(), args.dry_run, args.verbose))
    except KeyboardInterrupt:
        Console().print("\n[yellow]Interrumpido.[/]")
        exit_code = 130

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
