"""Architecture guards for platform-neutral Codex control."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _imports(path: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def test_discord_adapter_does_not_import_telegram_adapter() -> None:
    imports = _imports("plugins/platforms/discord/adapter.py")

    assert "gateway.platforms.telegram" not in imports
    assert "plugins.platforms.telegram.adapter" not in imports


def test_telegram_adapter_does_not_import_discord_adapter() -> None:
    imports = _imports("gateway/platforms/telegram.py")

    assert "plugins.platforms.discord.adapter" not in imports


def test_slash_codex_entry_uses_platform_neutral_service() -> None:
    imports = _imports("gateway/slash_commands.py")

    assert "gateway.control_planes.codex" in imports
    assert "tools.codex_app_server" not in imports
