"""Runtime configuration helpers for Claude command execution."""

from __future__ import annotations

import math
import os
from typing import Any


PLAN_PROMPT_PREFIX = (
    "Create a detailed implementation plan first. Include the files to change, "
    "the intended edits, dependencies between edits, tests, and acceptance "
    "criteria. Do not modify files or execute the implementation until the user "
    "confirms.\n\n"
)


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4.5"
DEFAULT_CLAUDE_PERMISSION_MODE = "acceptEdits"
DEFAULT_CLAUDE_EFFORT = "high"


def read_claude_config_model() -> str:
    """Read model from ~/.claude/settings.json, falling back to a default."""
    try:
        import json

        path = os.path.expanduser("~/.claude/settings.json")
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        model = str(data.get("model") or "")
        return model or DEFAULT_CLAUDE_MODEL
    except Exception:
        return DEFAULT_CLAUDE_MODEL


def load_claude_cfg() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        value = cfg.get("claude_cli", {}) if isinstance(cfg, dict) else {}
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def normalize_permission_mode(value: str) -> str:
    raw = (value or "").strip()
    aliases = {
        "acceptedits": "acceptEdits",
        "accept_edits": "acceptEdits",
        "auto": "auto",
        "plan": "plan",
        "default": "default",
        "dontask": "dontAsk",
        "dont_ask": "dontAsk",
        "bypasspermissions": "bypassPermissions",
        "bypass_permissions": "bypassPermissions",
        "yolo": "bypassPermissions",
        "full": "bypassPermissions",
    }
    if not raw:
        return DEFAULT_CLAUDE_PERMISSION_MODE
    return aliases.get(raw.lower(), raw)


def claude_permission_profiles() -> dict[str, dict[str, str]]:
    """Claude permission mode profiles mirroring codex permission profiles."""
    return {
        "default": {
            "label": "默认",
            "permission_mode": "acceptEdits",
        },
        "auto_review": {
            "label": "自动审批",
            "permission_mode": "auto",
        },
        "read_only": {
            "label": "只读",
            "permission_mode": "plan",
        },
        "full_access": {
            "label": "完全访问",
            "permission_mode": "bypassPermissions",
        },
    }


def normalize_permission_profile(value: str) -> str:
    raw = (value or "").strip().lower().replace("_", "-")
    aliases = {
        "": "default",
        "default": "default",
        "workspace": "default",
        "safe": "default",
        "accept-edits": "default",
        "auto": "auto_review",
        "auto-review": "auto_review",
        "auto-reviewer": "auto_review",
        "approve": "auto_review",
        "approve-for-me": "auto_review",
        "approveforme": "auto_review",
        "read": "read_only",
        "readonly": "read_only",
        "read-only": "read_only",
        "read-only-mode": "read_only",
        "plan": "read_only",
        "full": "full_access",
        "full-access": "full_access",
        "danger": "full_access",
        "danger-full-access": "full_access",
        "yolo": "full_access",
    }
    return aliases.get(raw, "")


def claude_cli_config_overrides(cfg: dict[str, Any] | None = None) -> list[str]:
    """Return per-session config overrides for Claude CLI subprocess.

    Unlike codex app-server, Claude CLI does not take config override flags.
    This returns an empty list but is kept for API parity with
    :func:`codex_app_server_config_overrides` so the session pool can compare
    configs for session retirement.
    """
    del cfg
    return []


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed) or parsed <= 0:
        return default
    return parsed


def claude_cli_turn_options(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Timeouts for a single Claude CLI turn.

    Long-running remote Claude tasks commonly exceed the transport default.
    Keep the values configurable from Hermes while preserving the transport's
    stream-idle watchdog behavior.
    """
    claude_cfg = cfg if isinstance(cfg, dict) else load_claude_cfg()
    turn_timeout = claude_cfg.get("turn_timeout_seconds", claude_cfg.get("turn_timeout", 1800.0))
    idle_timeout = claude_cfg.get(
        "idle_timeout_seconds",
        claude_cfg.get("stream_idle_timeout_seconds", 600.0),
    )
    return {
        "turn_timeout": _positive_float(turn_timeout, 1800.0),
        "idle_timeout": _positive_float(idle_timeout, 600.0),
    }


def get_approval_callback():
    try:
        from tools.terminal_tool import _get_approval_callback

        return _get_approval_callback()
    except Exception:
        return None
