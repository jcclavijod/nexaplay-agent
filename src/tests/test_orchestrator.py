"""Tests for src/agent/orchestrator.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(text: str) -> MagicMock:
    content = MagicMock()
    content.text = text
    resp = MagicMock()
    resp.content = [content]
    return resp


def _plan_json(steps: list[dict], goal: str = "Test goal") -> str:
    return json.dumps({
        "goal": goal,
        "success_criterion": "Completed",
        "steps": steps,
    })


def _make_hub(tool_responses: list[str]) -> MagicMock:
    hub = MagicMock()
    hub.list_all_tools = AsyncMock(return_value=[
        {
            "name": "nexaplay_api_call",
            "description": "Call NexaPlay API",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "code_generator",
            "description": "Generate code",
            "input_schema": {"type": "object", "properties": {}},
        },
    ])
    hub.call = AsyncMock(side_effect=tool_responses)
    return hub


def _make_client(*texts: str) -> MagicMock:
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=[_mock_response(t) for t in texts])
    return client


_GET_RESPONSE = json.dumps({
    "success": True,
    "data": {
        "service_id": "42",
        "client_id": "MX-01",
        "country": "MX",
        "operational_limit": 92,
        "min_allowed": 95,
        "standard_value": 96,
    },
})

_POST_RESPONSE = json.dumps({
    "success": True,
    "data": {
        "service_id": "42",
        "client_id": "MX-01",
        "country": "MX",
        "previous": {"operational_limit": 92},
        "updated": {"operational_limit": 95},
        "timestamp": "2024-01-01T00:00:00Z",
        "version": 2,
    },
})

_CODEGEN_RESPONSE = json.dumps({
    "success": True,
    "code": "def foo(): pass",
    "test": "def test_foo(): assert foo() is None",
    "filename": "service42.py",
})

_SUMMARY_TEXT = "El límite operacional del servicio 42 fue actualizado de 92 a 95."


# ---------------------------------------------------------------------------
# Full 3-step flow: GET → POST → codegen
# ---------------------------------------------------------------------------

async def test_full_3step_flow_executes(tmp_path, monkeypatch):
    """All three steps run; artifacts are written; status is completed."""
    monkeypatch.chdir(tmp_path)

    plan_steps = [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Read config",
            "inputs": {"method": "GET", "endpoint": "/services/42/config"},
            "precondition": None,
        },
        {
            "id": 2,
            "tool": "nexaplay_api_call",
            "purpose": "Update config",
            "inputs": {
                "method": "POST",
                "endpoint": "/services/42/config",
                "body": {"operational_limit": 95},
            },
            "precondition": None,
        },
        {
            "id": 3,
            "tool": "code_generator",
            "purpose": "Generate helper code",
            "inputs": {"filename": "service42.py"},
            "precondition": None,
        },
    ]

    hub = _make_hub([_GET_RESPONSE, _POST_RESPONSE, _CODEGEN_RESPONSE])
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)
    monkeypatch.setattr("builtins.input", lambda _: "confirmar")

    orch = Orchestrator(hub, client, max_iterations=15)
    result = await orch.run("Ajustar servicio 42", job_id="testjob001")

    assert result["status"] == "completed"
    assert result["job_id"] == "testjob001"

    # All three tools were invoked in order
    assert hub.call.call_count == 3
    call_tools = [c.args[0] for c in hub.call.call_args_list]
    assert call_tools == ["nexaplay_api_call", "nexaplay_api_call", "code_generator"]

    # Summary is present and non-empty
    assert result["summary"]
    assert not result["summary"].startswith("[UNVALIDATED]")

    # Artifacts written to workspace
    assert len(result["artifacts"]) == 2
    for path_str in result["artifacts"]:
        assert Path(path_str).exists()

    # Observations captured
    assert len(result["observations"]) == 3


# ---------------------------------------------------------------------------
# POST confirmation
# ---------------------------------------------------------------------------

async def test_post_not_confirmed_aborts(tmp_path, monkeypatch):
    """When user doesn't type 'confirmar', the POST is never called and status is aborted."""
    monkeypatch.chdir(tmp_path)

    plan_steps = [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Update config",
            "inputs": {
                "method": "POST",
                "endpoint": "/services/42/config",
                "body": {"operational_limit": 95},
            },
            "precondition": None,
        },
    ]

    hub = _make_hub([_POST_RESPONSE])
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)
    monkeypatch.setattr("builtins.input", lambda _: "no")

    orch = Orchestrator(hub, client)
    result = await orch.run("Update limit", job_id="testjob002")

    assert result["status"] == "aborted"
    # POST must not have been executed
    hub.call.assert_not_called()


async def test_post_confirmed_executes(tmp_path, monkeypatch):
    """When user types 'confirmar', the POST is executed and status is completed."""
    monkeypatch.chdir(tmp_path)

    plan_steps = [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Update config",
            "inputs": {
                "method": "POST",
                "endpoint": "/services/42/config",
                "body": {"operational_limit": 95},
            },
            "precondition": None,
        },
    ]

    hub = _make_hub([_POST_RESPONSE])
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)
    monkeypatch.setattr("builtins.input", lambda _: "confirmar")

    orch = Orchestrator(hub, client)
    result = await orch.run("Update limit", job_id="testjob003")

    assert result["status"] == "completed"
    hub.call.assert_called_once()
    assert hub.call.call_args.args[0] == "nexaplay_api_call"


async def test_post_confirmation_whitespace_trimmed(tmp_path, monkeypatch):
    """'confirmar' with surrounding whitespace is still accepted."""
    monkeypatch.chdir(tmp_path)

    plan_steps = [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Update",
            "inputs": {"method": "POST", "endpoint": "/services/42/config", "body": {}},
            "precondition": None,
        },
    ]

    hub = _make_hub([_POST_RESPONSE])
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)
    monkeypatch.setattr("builtins.input", lambda _: "  confirmar  ")

    result = await Orchestrator(hub, client).run("Update", job_id="testjob004")
    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# Precondition: step skipped when False
# ---------------------------------------------------------------------------

async def test_precondition_false_skips_step(tmp_path, monkeypatch):
    """A step whose precondition evaluates to False is skipped without calling the tool."""
    monkeypatch.chdir(tmp_path)

    # Step 1 returns operational_limit=97 (already above min_allowed=95)
    get_above_limit = json.dumps({
        "success": True,
        "data": {"service_id": "42", "operational_limit": 97, "min_allowed": 95},
    })

    plan_steps = [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Read config",
            "inputs": {"method": "GET", "endpoint": "/services/42/config"},
            "precondition": None,
        },
        {
            "id": 2,
            "tool": "nexaplay_api_call",
            "purpose": "Update (only if below min)",
            "inputs": {
                "method": "POST",
                "endpoint": "/services/42/config",
                "body": {"operational_limit": 95},
            },
            # Precondition is False because 97 >= 95
            "precondition": "$step1.data.operational_limit < $step1.data.min_allowed",
        },
    ]

    hub = _make_hub([get_above_limit])  # only 1 response needed
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)
    monkeypatch.setattr("builtins.input", lambda _: "confirmar")

    result = await Orchestrator(hub, client).run("Fix limit if needed", job_id="testjob005")

    assert result["status"] == "completed"
    # Only the GET was called; POST was skipped
    assert hub.call.call_count == 1


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

async def test_loop_detection_aborts_after_3_identical_actions(tmp_path, monkeypatch):
    """Three consecutive identical GET calls trigger loop detection and abort."""
    monkeypatch.chdir(tmp_path)

    same_step = {
        "id": 1,
        "tool": "nexaplay_api_call",
        "purpose": "Read config",
        "inputs": {"method": "GET", "endpoint": "/services/42/config"},
        "precondition": None,
    }

    plan_steps = [
        {**same_step, "id": 1},
        {**same_step, "id": 2},
        {**same_step, "id": 3},
    ]

    # We need 3 GET responses available (loop triggers *after* the 3rd step is resolved,
    # before calling the tool — but the check happens on the resolved inputs, which differ
    # by step.id injected later). Actually loop detection uses resolved inputs BEFORE
    # injection, so all three are {"method": "GET", "endpoint": "..."} and ARE identical.
    hub = _make_hub([_GET_RESPONSE] * 3)
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)

    result = await Orchestrator(hub, client).run("Repeat GET", job_id="testjob006")

    assert result["status"] == "aborted"
    # At most 2 calls made before loop detected on 3rd
    assert hub.call.call_count <= 2


# ---------------------------------------------------------------------------
# Max iterations
# ---------------------------------------------------------------------------

async def test_max_iterations_respected(tmp_path, monkeypatch):
    """With max_iterations=2, a 5-step plan is cut off after 2 steps."""
    monkeypatch.chdir(tmp_path)

    plan_steps = [
        {
            "id": i,
            "tool": "nexaplay_api_call",
            "purpose": f"Read step {i}",
            "inputs": {"method": "GET", "endpoint": f"/services/{i}/config"},
            "precondition": None,
        }
        for i in range(1, 6)
    ]

    hub = _make_hub([_GET_RESPONSE] * 5)
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)

    result = await Orchestrator(hub, client, max_iterations=2).run(
        "Multi-step", job_id="testjob007"
    )

    assert result["status"] == "aborted"
    assert hub.call.call_count == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

async def test_silent_write_failure_aborts_immediately(tmp_path, monkeypatch):
    """SILENT_WRITE_FAILURE in tool response causes immediate abort, no further steps."""
    monkeypatch.chdir(tmp_path)

    swf_response = json.dumps({
        "success": False,
        "error": "SILENT_WRITE_FAILURE",
        "message": "Response 2xx but value unchanged",
    })

    plan_steps = [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Update config",
            "inputs": {"method": "GET", "endpoint": "/services/42/config"},
            "precondition": None,
        },
        {
            "id": 2,
            "tool": "nexaplay_api_call",
            "purpose": "This should not run",
            "inputs": {"method": "GET", "endpoint": "/services/42/config"},
            "precondition": None,
        },
    ]

    hub = _make_hub([swf_response, _GET_RESPONSE])
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)

    result = await Orchestrator(hub, client).run("Update", job_id="testjob008")

    assert result["status"] == "aborted"
    # Second step must not have been called
    assert hub.call.call_count == 1


async def test_tool_error_aborts(tmp_path, monkeypatch):
    """Any tool error (NETWORK_ERROR, SERVER_ERROR) causes abort."""
    monkeypatch.chdir(tmp_path)

    error_response = json.dumps({
        "success": False,
        "error": "SERVER_ERROR",
        "message": "5xx after retries",
    })

    plan_steps = [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Read config",
            "inputs": {"method": "GET", "endpoint": "/services/42/config"},
            "precondition": None,
        },
    ]

    hub = _make_hub([error_response])
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)

    result = await Orchestrator(hub, client).run("Update", job_id="testjob009")

    assert result["status"] == "aborted"


# ---------------------------------------------------------------------------
# Summary tagged [UNVALIDATED] when empty
# ---------------------------------------------------------------------------

async def test_empty_summary_tagged_unvalidated(tmp_path, monkeypatch):
    """An empty summary response is prefixed with [UNVALIDATED]."""
    monkeypatch.chdir(tmp_path)

    plan_steps = [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Read config",
            "inputs": {"method": "GET", "endpoint": "/services/42/config"},
            "precondition": None,
        },
    ]

    hub = _make_hub([_GET_RESPONSE])
    client = _make_client(_plan_json(plan_steps), "")  # empty summary

    result = await Orchestrator(hub, client).run("Check config", job_id="testjob010")

    assert result["summary"].startswith("[UNVALIDATED]")


# ---------------------------------------------------------------------------
# Secondary SILENT_WRITE_FAILURE (verify_post_applied)
# ---------------------------------------------------------------------------

async def test_post_silent_write_failure_secondary_check(tmp_path, monkeypatch):
    """POST response 2xx but updated value doesn't match → SILENT_WRITE_FAILURE."""
    monkeypatch.chdir(tmp_path)

    # Server says success but updated.operational_limit is still 92, not 95
    bad_post = json.dumps({
        "success": True,
        "data": {
            "previous": {"operational_limit": 92},
            "updated": {"operational_limit": 92},  # ← mismatch
        },
    })

    plan_steps = [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Update config",
            "inputs": {
                "method": "POST",
                "endpoint": "/services/42/config",
                "body": {"operational_limit": 95},
            },
            "precondition": None,
        },
    ]

    hub = _make_hub([bad_post])
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)
    monkeypatch.setattr("builtins.input", lambda _: "confirmar")

    result = await Orchestrator(hub, client).run("Update", job_id="testjob011")

    assert result["status"] == "aborted"


# ---------------------------------------------------------------------------
# job_id and step injected into tool arguments
# ---------------------------------------------------------------------------

async def test_job_id_and_step_injected(tmp_path, monkeypatch):
    """job_id and step are present in the arguments passed to hub.call."""
    monkeypatch.chdir(tmp_path)

    plan_steps = [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Read config",
            "inputs": {"method": "GET", "endpoint": "/services/42/config"},
            "precondition": None,
        },
    ]

    hub = _make_hub([_GET_RESPONSE])
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)

    result = await Orchestrator(hub, client).run("Check", job_id="myjob42")

    _, call_kwargs = hub.call.call_args
    call_args_positional = hub.call.call_args.args
    passed_arguments = call_args_positional[1]  # second positional arg = arguments dict

    assert passed_arguments["job_id"] == "myjob42"
    assert passed_arguments["step"] == 1


# ---------------------------------------------------------------------------
# GET is autonomous (no confirmation prompt)
# ---------------------------------------------------------------------------

async def test_get_requires_no_confirmation(tmp_path, monkeypatch):
    """GET calls proceed without triggering input()."""
    monkeypatch.chdir(tmp_path)

    plan_steps = [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Read config",
            "inputs": {"method": "GET", "endpoint": "/services/42/config"},
            "precondition": None,
        },
    ]

    hub = _make_hub([_GET_RESPONSE])
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)

    input_called = []
    monkeypatch.setattr("builtins.input", lambda _: input_called.append(True) or "")

    result = await Orchestrator(hub, client).run("Check", job_id="testjob012")

    assert result["status"] == "completed"
    assert not input_called  # input() was never called


# ---------------------------------------------------------------------------
# console.print called when console is provided
# ---------------------------------------------------------------------------

async def test_console_print_called_when_provided(tmp_path, monkeypatch):
    """When console is passed, console.print is used for output."""
    monkeypatch.chdir(tmp_path)

    plan_steps = [
        {
            "id": 1,
            "tool": "nexaplay_api_call",
            "purpose": "Read config",
            "inputs": {"method": "GET", "endpoint": "/services/42/config"},
            "precondition": None,
        },
    ]

    hub = _make_hub([_GET_RESPONSE])
    client = _make_client(_plan_json(plan_steps), _SUMMARY_TEXT)

    console = MagicMock()
    await Orchestrator(hub, client, console=console).run("Check", job_id="testjob013")

    assert console.print.called


# ---------------------------------------------------------------------------
# Unhandled exception returns error dict
# ---------------------------------------------------------------------------

async def test_unhandled_exception_returns_error_dict(tmp_path, monkeypatch):
    """If list_all_tools raises unexpectedly, run() returns an error dict."""
    monkeypatch.chdir(tmp_path)

    hub = MagicMock()
    hub.list_all_tools = AsyncMock(side_effect=RuntimeError("Network gone"))
    client = MagicMock()

    result = await Orchestrator(hub, client).run("anything", job_id="testjob014")

    assert result["status"] == "error"
    assert "Network gone" in result["error"]
    assert result["job_id"] == "testjob014"
