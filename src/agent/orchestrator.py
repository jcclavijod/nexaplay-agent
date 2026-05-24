"""Orchestrator: Plan → ReAct → Validation → Summary cycle."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent import planner, validator
from src.agent.context_manager import ContextManager, Observation
from src.agent.mcp_client import MCPHub
from src.agent.model_router import TaskType, get_max_tokens, get_model, get_temperature
from src.agent.planner import Plan, PlanStep

_PROMPTS_DIR = Path(__file__).parent / "prompts"
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Retorna la fecha y hora actual en formato ISO 8601 UTC."""
    return datetime.now(timezone.utc).isoformat()


class Orchestrator:
    """Coordina el ciclo Plan → ReAct → Validación → Resumen para un job.

    Recibe un requerimiento en lenguaje natural, genera un plan con Claude,
    ejecuta cada paso invocando tools a través del :class:`MCPHub` y produce
    un resumen estructurado del resultado.

    Los métodos ``_on_step_start``, ``_on_step_result``, ``_request_post_confirmation``
    y ``_call_tool`` son hooks sobreescribibles para personalizar la presentación
    o interceptar llamadas (ver :class:`~src.agent.cli.RichOrchestrator`).

    Ejemplo::

        async with MCPHub() as hub:
            orch = Orchestrator(hub, anthropic_client)
            result = await orch.run("Listar los 5 servicios más activos")
    """

    def __init__(
        self,
        hub: MCPHub,
        anthropic_client: Any,
        max_iterations: int = 15,
        console: Any = None,
    ) -> None:
        """Inicializa el orchestrator.

        Args:
            hub: Hub de MCP con los servers conectados.
            anthropic_client: Cliente ``anthropic.AsyncAnthropic`` para llamadas al modelo.
            max_iterations: Límite de pasos ejecutados antes de abortar el job.
            console: Objeto con método ``print`` para output de progreso. ``None`` silencia
                el output de texto.
        """
        self._hub = hub
        self._client = anthropic_client
        self._max_iterations = max_iterations
        self._console = console

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    async def run(self, requirement: str, job_id: str | None = None) -> dict:
        """Ejecuta el ciclo completo para *requirement* y retorna el resultado del job.

        Crea el workspace del job, ejecuta :meth:`_execute` y captura cualquier
        excepción no manejada para devolverla como resultado con ``status="error"``.

        Args:
            requirement: Descripción del objetivo en lenguaje natural.
            job_id: Identificador del job. Si es ``None``, se genera un UUID hex.

        Returns:
            Dict con las keys ``job_id``, ``status``, ``plan``, ``observations``,
            ``summary`` y ``artifacts``. En caso de error: ``job_id``, ``status``
            y ``error``.
        """
        job_id = job_id or uuid.uuid4().hex
        workspace = Path("workspace") / job_id
        workspace.mkdir(parents=True, exist_ok=True)

        try:
            return await self._execute(requirement, job_id, workspace)
        except Exception as exc:
            logger.exception("Unhandled error in job %s", job_id)
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(exc),
                "partial_state": {},
            }

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def _execute(self, requirement: str, job_id: str, workspace: Path) -> dict:
        """Núcleo de ejecución: planifica y ejecuta cada paso del plan.

        Args:
            requirement: Requerimiento a satisfacer.
            job_id: Identificador único del job en curso.
            workspace: Directorio de trabajo donde se escriben los artefactos.

        Returns:
            Dict completo con estado, plan, observaciones, resumen y artefactos.
        """
        ctx = ContextManager()
        step_results: dict[int, dict] = {}
        artifacts: list[str] = []
        recent_actions: list[tuple[str, str]] = []

        # --- Plan -------------------------------------------------------
        tools = await self._hub.list_all_tools()
        plan: Plan = await planner.create_plan(requirement, tools, self._client)
        self._print(self._format_plan_for_display(plan))
        logger.info(
            "job=%s action=plan_created goal=%r steps=%d",
            job_id,
            plan.goal,
            len(plan.steps),
        )

        iterations = 0
        aborted = False
        abort_reason = ""

        for step in plan.steps:
            if iterations >= self._max_iterations:
                abort_reason = f"Max iterations ({self._max_iterations}) reached"
                aborted = True
                logger.warning("job=%s step=%d %s", job_id, step.id, abort_reason)
                break

            iterations += 1

            # --- Precondition -------------------------------------------
            try:
                precond_ok = planner.evaluate_precondition(step.precondition, step_results)
            except ValueError as exc:
                logger.warning(
                    "job=%s step=%d action=precondition_error error=%r",
                    job_id,
                    step.id,
                    str(exc),
                )
                precond_ok = True  # conservative: attempt step when precondition is unresolvable

            if not precond_ok:
                logger.info(
                    "job=%s step=%d action=step_skipped reason=precondition",
                    job_id,
                    step.id,
                )
                self._print(f"[paso {step.id}] Omitido por precondición.")
                continue

            # --- Resolve inputs -----------------------------------------
            try:
                resolved = planner.resolve_inputs(step.inputs, step_results)
            except ValueError as exc:
                abort_reason = f"Input resolution failed at step {step.id}: {exc}"
                aborted = True
                logger.error("job=%s step=%d error=%r", job_id, step.id, abort_reason)
                break

            # --- Loop detection (checked before job_id/step injection) --
            action_key = (step.tool, json.dumps(resolved, sort_keys=True))
            recent_actions.append(action_key)
            if len(recent_actions) > 3:
                recent_actions.pop(0)
            if len(recent_actions) == 3 and len(set(recent_actions)) == 1:
                abort_reason = (
                    f"Loop detectado: 3 acciones idénticas consecutivas (tool={step.tool})"
                )
                aborted = True
                logger.error("job=%s step=%d action=loop_detected", job_id, step.id)
                break

            # --- POST confirmation ----------------------------------------
            method = str(resolved.get("method") or "").upper()
            if step.tool == "nexaplay_api_call" and method == "POST":
                if not self._request_post_confirmation(step.id, resolved):
                    abort_reason = f"Usuario no confirmó el POST en paso {step.id}"
                    aborted = True
                    logger.warning(
                        "job=%s step=%d action=post_not_confirmed",
                        job_id,
                        step.id,
                    )
                    break

            # --- Inject job_id / step ------------------------------------
            resolved["job_id"] = job_id
            resolved["step"] = step.id

            # --- Call tool -----------------------------------------------
            self._on_step_start(step, resolved)
            logger.info(
                "job=%s step=%d tool=%r action=call ts=%s",
                job_id,
                step.id,
                step.tool,
                _now_iso(),
            )
            raw_result = await self._call_tool(step.tool, resolved)

            # --- Parse result -------------------------------------------
            try:
                result_data: dict = json.loads(raw_result)
            except json.JSONDecodeError:
                result_data = {"success": False, "error": "PARSE_ERROR", "raw": raw_result}

            if step.tool == "code_generator":  # DEBUG_TEMP
                logger.debug(  # DEBUG_TEMP
                    "CODEGEN_ORCHESTRATOR_RESULT keys=%s dump=%r",  # DEBUG_TEMP
                    list(result_data.keys()),  # DEBUG_TEMP
                    {k: str(v)[:200] for k, v in result_data.items()},  # DEBUG_TEMP
                )  # DEBUG_TEMP

            # --- Observation --------------------------------------------
            obs = Observation(
                step_id=step.id,
                tool_name=step.tool,
                arguments=resolved,
                result=result_data,
            )
            await ctx.add_observation(obs)
            step_results[step.id] = result_data
            self._on_step_result(step, resolved, result_data)

            logger.info(
                "job=%s step=%d tool=%r status=%s ts=%s",
                job_id,
                step.id,
                step.tool,
                "ok" if result_data.get("success") else "fail",
                _now_iso(),
            )

            # --- Error handling -----------------------------------------
            if not result_data.get("success"):
                error_code = result_data.get("error", "UNKNOWN")
                if error_code == "SILENT_WRITE_FAILURE":
                    abort_reason = (
                        f"SILENT_WRITE_FAILURE en paso {step.id}: "
                        f"{result_data.get('message', 'sin detalle')}"
                    )
                    aborted = True
                    logger.error(
                        "job=%s step=%d action=silent_write_failure ts=%s",
                        job_id,
                        step.id,
                        _now_iso(),
                    )
                    self._print(
                        f"[FATAL] SILENT_WRITE_FAILURE en paso {step.id}. "
                        "Servidor respondió 2xx pero el cambio no se aplicó. "
                        "Abortando sin reintentos."
                    )
                else:
                    abort_reason = f"Tool error at step {step.id}: {error_code}"
                    aborted = True
                    logger.error(
                        "job=%s step=%d tool_error=%r ts=%s",
                        job_id,
                        step.id,
                        error_code,
                        _now_iso(),
                    )
                    self._print(
                        f"[paso {step.id}] Tool devolvió error: {error_code}. Abortando."
                    )
                break

            # --- POST secondary SILENT_WRITE_FAILURE check --------------
            if step.tool == "nexaplay_api_call" and method == "POST":
                body = resolved.get("body") or {}
                ok, swf_msg = validator.verify_post_applied(result_data, body)
                if not ok:
                    abort_reason = swf_msg or "SILENT_WRITE_FAILURE"
                    aborted = True
                    logger.error(
                        "job=%s step=%d action=silent_write_failure_secondary ts=%s",
                        job_id,
                        step.id,
                        _now_iso(),
                    )
                    self._print(f"[FATAL] {abort_reason}")
                    break

            # --- code_generator artefact --------------------------------
            if step.tool == "code_generator" and result_data.get("success"):
                code = result_data.get("code", "")
                test_code = result_data.get("test", "")
                filename = result_data.get("filename", f"step{step.id}_generated.py")

                code_path = workspace / filename
                test_path = workspace / f"test_{filename}"

                valid_code, reason_code = validator.validate_workspace_path(
                    str(code_path), job_id
                )
                valid_test, reason_test = validator.validate_workspace_path(
                    str(test_path), job_id
                )

                if not valid_code:
                    logger.warning(
                        "job=%s step=%d action=workspace_path_rejected reason=%r",
                        job_id,
                        step.id,
                        reason_code,
                    )
                elif not valid_test:
                    logger.warning(
                        "job=%s step=%d action=workspace_path_rejected reason=%r",
                        job_id,
                        step.id,
                        reason_test,
                    )
                else:
                    code_path.write_text(code, encoding="utf-8")
                    test_path.write_text(test_code, encoding="utf-8")
                    artifacts.extend([str(code_path), str(test_path)])
                    logger.info(
                        "job=%s step=%d action=artifacts_written files=%s",
                        job_id,
                        step.id,
                        [str(code_path), str(test_path)],
                    )
                    self._print(
                        f"[paso {step.id}] Artefactos escritos: {code_path}, {test_path}"
                    )

        # --- Summary ----------------------------------------------------
        summary = await self._generate_summary(
            job_id=job_id,
            plan=plan,
            observations=ctx.observations,
            step_results=step_results,
            aborted=aborted,
            abort_reason=abort_reason,
        )

        status = "aborted" if aborted else "completed"
        self._print(f"\n[Resumen]\n{summary}")

        return {
            "job_id": job_id,
            "status": status,
            "plan": plan.model_dump(),
            "observations": [o.model_dump() for o in ctx.observations],
            "summary": summary,
            "artifacts": artifacts,
        }

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _request_post_confirmation(self, step_id: int, resolved: dict) -> bool:
        """Solicita confirmación interactiva antes de ejecutar un POST.

        Imprime el endpoint y el body, luego espera que el usuario escriba
        ``"confirmar"`` para proceder. Sobreescribir para integrar con UIs
        alternativas (ver :class:`~src.agent.cli.RichOrchestrator`).

        Args:
            step_id: Número de paso del plan que genera el POST.
            resolved: Inputs ya resueltos del paso, incluyendo ``endpoint`` y ``body``.

        Returns:
            ``True`` si el usuario confirmó, ``False`` en caso contrario.
        """
        self._print(
            f"\n[paso {step_id}] POST a ejecutar:\n"
            f"  endpoint : {resolved.get('endpoint', '')}\n"
            f"  body     : {json.dumps(resolved.get('body', {}), indent=2, ensure_ascii=False)}\n"
        )
        confirmation = input("¿Confirmar esta escritura? Escribe 'confirmar': ")
        confirmed = confirmation.strip() == "confirmar"
        if not confirmed:
            logger.info("post_not_confirmed step=%d response=%r", step_id, confirmation)
        return confirmed

    def _on_step_start(self, step: PlanStep, resolved: dict) -> None:
        """Hook invocado justo antes de llamar a una tool. Sobreescribir para display por paso."""

    def _on_step_result(self, step: PlanStep, resolved: dict, result_data: dict) -> None:
        """Hook invocado tras parsear el resultado de una tool. Sobreescribir para display por paso."""

    async def _call_tool(self, tool_name: str, arguments: dict) -> str:
        """Invoca una tool a través del hub. Sobreescribir para interceptar o mockear llamadas.

        Args:
            tool_name: Nombre de la tool a invocar.
            arguments: Argumentos ya resueltos para la tool.

        Returns:
            Resultado crudo de la tool como string JSON.
        """
        return await self._hub.call(tool_name, arguments)

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    async def _generate_summary(
        self,
        job_id: str,
        plan: Plan,
        observations: list[Observation],
        step_results: dict[int, dict],
        aborted: bool,
        abort_reason: str,
    ) -> str:
        """Genera el resumen final del job llamando a Claude con el prompt ``summarizer.md``.

        Args:
            job_id: Identificador del job.
            plan: Plan ejecutado (o intentado).
            observations: Lista de observaciones activas al finalizar.
            step_results: Mapa de ``step_id → resultado`` de todos los pasos.
            aborted: ``True`` si el job terminó de forma anormal.
            abort_reason: Razón del aborto (ignorada si ``aborted=False``).

        Returns:
            Texto del resumen en español. Si la respuesta está vacía o es ilegible,
            se antepone el prefijo ``"[UNVALIDATED]"``.
        """
        prompt_template = (_PROMPTS_DIR / "summarizer.md").read_text(encoding="utf-8")

        trace = {
            "job_id": job_id,
            "status": "aborted" if aborted else "completed",
            "abort_reason": abort_reason if aborted else None,
            "plan": plan.model_dump(),
            "step_results": {str(k): v for k, v in step_results.items()},
            "observations": [o.model_dump() for o in observations],
        }

        prompt = prompt_template.replace(
            "{trace_json}", json.dumps(trace, ensure_ascii=False, indent=2)
        )

        model = get_model(TaskType.SUMMARIZATION)
        response = await self._client.messages.create(
            model=model,
            max_tokens=get_max_tokens(TaskType.SUMMARIZATION),
            temperature=get_temperature(TaskType.SUMMARIZATION),
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Two independent checks; [UNVALIDATED] only when both fail
        check1 = bool(raw and raw.strip())
        check2 = bool(raw and any(c.isalpha() for c in raw))
        if not check1 and not check2:
            return f"[UNVALIDATED] {raw}"
        return raw

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_plan_for_display(self, plan: Plan) -> str:
        """Formatea el plan como texto multilinea para mostrar en consola.

        Args:
            plan: Plan a formatear.

        Returns:
            String con el objetivo, criterio de éxito y lista de pasos numerados.
        """
        lines = [
            f"Plan: {plan.goal}",
            f"Criterio de éxito: {plan.success_criterion}",
            "",
        ]
        for s in plan.steps:
            precond = f" [si: {s.precondition}]" if s.precondition else ""
            lines.append(f"  [{s.id}] {s.tool}{precond}")
            lines.append(f"       Propósito: {s.purpose}")
        return "\n".join(lines)

    def _print(self, msg: str) -> None:
        """Envía *msg* a la consola si hay una configurada; de lo contrario no hace nada."""
        if self._console is not None:
            self._console.print(msg)
