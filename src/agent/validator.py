"""Agent-level output validators for NexaPlay.

Provides a second line of defense beyond the http_client checks:
  - Python code syntax validation via ast.parse
  - POST response consistency (SILENT_WRITE_FAILURE detection)
  - Workspace path confinement and symlink rejection
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any


def validate_python_code(code: str) -> tuple[bool, str | None]:
    """Retorna ``(True, None)`` si *code* es Python sintácticamente válido.

    Args:
        code: Código Python a validar.

    Returns:
        Tupla ``(válido, mensaje_error)``. Si es válido, el mensaje es ``None``.
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"


def verify_post_applied(
    post_response: dict[str, Any],
    expected_body: dict[str, Any],
) -> tuple[bool, str | None]:
    """Verifica que todos los campos de *expected_body* se reflejen en la respuesta POST.

    Comprueba que cada campo de *expected_body* aparezca en
    ``post_response["data"]["updated"]`` con el mismo valor.

    Args:
        post_response: Respuesta JSON completa del endpoint POST.
        expected_body: Dict con los campos y valores que se esperaban aplicar.

    Returns:
        Tupla ``(ok, mensaje)``. Si falla, el mensaje incluye el prefijo
        ``"SILENT_WRITE_FAILURE:"`` con el detalle del campo discrepante.
    """
    updated: dict[str, Any] = (
        (post_response.get("data") or {}).get("updated") or {}
    )

    for field, expected_value in expected_body.items():
        actual = updated.get(field)
        if actual != expected_value:
            return (
                False,
                f"SILENT_WRITE_FAILURE: campo '{field}' esperaba {expected_value!r}, "
                f"recibió {actual!r}",
            )

    return True, None


def validate_workspace_path(
    path: str, job_id: str
) -> tuple[bool, str | None]:
    """Verifica que *path* esté confinado en ``./workspace/{job_id}/``.

    Rechaza si:

    - el path contiene un componente ``..`` (traversal antes de la resolución)
    - el path absoluto resuelto escapa del workspace del job
    - cualquier componente desde el workspace hacia abajo es un symlink

    Args:
        path: Ruta a validar (relativa o absoluta).
        job_id: Identificador del job que delimita el workspace permitido.

    Returns:
        Tupla ``(válido, mensaje_error)``. Si es válido, el mensaje es ``None``.
    """
    # 1. Reject ".." before any resolution — guard against obvious traversal.
    if ".." in Path(path).parts:
        return False, f"Path traversal rejected: '..' component in '{path}'"

    # 2. Compute the workspace boundary as an absolute path (no symlink follow).
    workspace_base = Path(os.path.abspath(os.path.join("workspace", job_id)))

    # 3. Resolve the candidate path the same way (abspath, no symlink follow).
    abs_candidate = Path(os.path.abspath(path))

    # 4. Confinement check.
    try:
        abs_candidate.relative_to(workspace_base)
    except ValueError:
        return (
            False,
            f"Path '{path}' resolves outside workspace for job '{job_id}'",
        )

    # 5. Symlink check — walk from candidate up to (and including) workspace_base.
    cursor = abs_candidate
    while True:
        if cursor.is_symlink():
            return False, f"Symlink rejected at '{cursor}'"
        if cursor == workspace_base:
            break
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent

    return True, None
