"""Runtime configuration helpers for Codex command execution."""

from __future__ import annotations

import os
from typing import Any


PLAN_PROMPT_PREFIX = (
    "Create a detailed implementation plan first. Include the files to change, "
    "the intended edits, dependencies between edits, tests, and acceptance "
    "criteria. Do not modify files or execute the implementation until the user "
    "confirms.\n\n"
)


def read_codex_config_model() -> str:
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib
    try:
        with open(os.path.expanduser("~/.codex/config.toml"), "rb") as handle:
            return str(tomllib.load(handle).get("model", "gpt-5.5"))
    except Exception:
        return "gpt-5.5"


def load_codex_cfg() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        value = cfg.get("codex_app_server", {}) if isinstance(cfg, dict) else {}
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def get_approval_callback():
    try:
        from tools.terminal_tool import _get_approval_callback

        return _get_approval_callback()
    except Exception:
        return None
