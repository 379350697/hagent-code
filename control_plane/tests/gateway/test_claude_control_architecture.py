"""Architecture guards for platform-neutral Claude control."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HERMES_ROOT = Path(os.environ.get("HERMES_AGENT", ROOT)).resolve()


def _imports(path: str, *, root: Path = ROOT) -> set[str]:
    target = root / path
    if not target.exists():
        pytest.skip(f"{path} is only available in a full Hermes checkout")
    tree = ast.parse(target.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def test_claude_control_plane_does_not_import_legacy_tool() -> None:
    for path in (ROOT / "gateway/control_planes/claude").glob("*.py"):
        imports = _imports(str(path.relative_to(ROOT)))

        assert "tools.codex_app_server" not in imports
        assert "tools.claude_app_server" not in imports


def test_claude_transport_does_not_import_codex_transport() -> None:
    imports = _imports("agent/transports/claude_cli_session.py")

    assert "agent.transports.codex_app_server_session" not in imports
    assert "agent.transports.codex_app_server" not in imports


def test_slash_claude_entry_uses_platform_neutral_service() -> None:
    # The slash_commands.py overlay lives in ../hermes_overlay/ (one level
    # above control_plane/). Prefer the overlay copy when present so this
    # test passes before the overlay is installed into a Hermes checkout.
    overlay = ROOT.parent / "hermes_overlay" / "gateway" / "slash_commands.py"
    if overlay.exists():
        tree = ast.parse(overlay.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.add(node.module)
    else:
        names = _imports("gateway/slash_commands.py", root=HERMES_ROOT)

    assert "gateway.control_planes.claude" in names
