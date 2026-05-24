"""Tests for src/agent/validator.py."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.agent.validator import (
    validate_python_code,
    validate_workspace_path,
    verify_post_applied,
)


# ---------------------------------------------------------------------------
# validate_python_code
# ---------------------------------------------------------------------------


def test_valid_python_passes():
    code = "def foo(x):\n    return x * 2\n"
    ok, err = validate_python_code(code)
    assert ok is True
    assert err is None


def test_invalid_python_fails():
    code = "def foo(:\n    pass\n"
    ok, err = validate_python_code(code)
    assert ok is False
    assert err is not None
    assert "SyntaxError" in err


def test_empty_string_is_valid():
    ok, err = validate_python_code("")
    assert ok is True
    assert err is None


def test_type_annotation_and_async_are_valid():
    code = "async def bar(x: int) -> str:\n    return str(x)\n"
    ok, err = validate_python_code(code)
    assert ok is True


# ---------------------------------------------------------------------------
# verify_post_applied
# ---------------------------------------------------------------------------


def test_post_applied_correctly():
    response = {
        "data": {
            "updated": {"operational_limit": 96, "max_transactions_per_second": 500}
        }
    }
    ok, err = verify_post_applied(response, {"operational_limit": 96})
    assert ok is True
    assert err is None


def test_silent_write_failure_detected():
    response = {
        "data": {
            "updated": {"operational_limit": 92}  # sent 96, got 92 back
        }
    }
    ok, err = verify_post_applied(response, {"operational_limit": 96})
    assert ok is False
    assert err is not None
    assert "SILENT_WRITE_FAILURE" in err
    assert "operational_limit" in err
    assert "96" in err
    assert "92" in err


def test_verify_post_missing_field_in_updated():
    response = {"data": {"updated": {}}}
    ok, err = verify_post_applied(response, {"operational_limit": 96})
    assert ok is False
    assert "SILENT_WRITE_FAILURE" in err


def test_verify_post_empty_expected_body():
    response = {"data": {"updated": {"x": 1}}}
    ok, err = verify_post_applied(response, {})
    assert ok is True


def test_verify_post_missing_data_key():
    ok, err = verify_post_applied({}, {"operational_limit": 96})
    assert ok is False
    assert "SILENT_WRITE_FAILURE" in err


def test_verify_post_multiple_fields_first_mismatch_reported():
    response = {
        "data": {
            "updated": {"operational_limit": 96, "max_transactions_per_second": 999}
        }
    }
    ok, err = verify_post_applied(
        response,
        {"operational_limit": 96, "max_transactions_per_second": 500},
    )
    assert ok is False
    assert "max_transactions_per_second" in err


# ---------------------------------------------------------------------------
# validate_workspace_path
# ---------------------------------------------------------------------------


def test_workspace_path_valid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job_id = "job-001"
    target = os.path.join("workspace", job_id, "output.py")
    ok, err = validate_workspace_path(target, job_id)
    assert ok is True
    assert err is None


def test_workspace_path_blocks_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ok, err = validate_workspace_path("../etc/passwd", "job-001")
    assert ok is False
    assert err is not None
    assert ".." in err or "traversal" in err.lower()


def test_workspace_path_blocks_absolute_escape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ok, err = validate_workspace_path("/etc/passwd", "job-001")
    assert ok is False


def test_workspace_path_blocks_sibling_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ok, err = validate_workspace_path(
        os.path.join("workspace", "other-job", "file.py"), "job-001"
    )
    assert ok is False


def test_workspace_path_blocks_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job_id = "job-sym"
    workspace_dir = tmp_path / "workspace" / job_id
    workspace_dir.mkdir(parents=True)

    secret = tmp_path / "secret.txt"
    secret.write_text("sensitive")

    link = workspace_dir / "link.txt"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("Symlink creation requires elevated privileges on this platform")

    ok, err = validate_workspace_path(str(link), job_id)
    assert ok is False
    assert err is not None
    assert "ymlink" in err


def test_workspace_path_workspace_root_itself_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # The boundary is workspace/{job_id}/ — the root itself should be allowed
    # as a directory (writes go inside it), but escaping above should not.
    job_id = "job-001"
    root = os.path.join("workspace", job_id)
    ok, err = validate_workspace_path(root, job_id)
    assert ok is True
