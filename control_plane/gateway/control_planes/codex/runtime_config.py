"""Runtime configuration helpers for Codex command execution."""

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


def normalize_sandbox_mode(value: str) -> str:
    raw = (value or "").strip().lower()
    aliases = {
        "workspace": "workspace-write",
        "workspace_write": "workspace-write",
        "workspace-write": "workspace-write",
        "read": "read-only",
        "readonly": "read-only",
        "read-only": "read-only",
        "danger": "danger-full-access",
        "danger-full-access": "danger-full-access",
        "full": "danger-full-access",
        "full-access": "danger-full-access",
        "yolo": "danger-full-access",
    }
    return aliases.get(raw, "workspace-write")


def codex_permission_profiles() -> dict[str, dict[str, str]]:
    """Desktop-aligned Codex permission profiles."""
    return {
        "default": {
            "label": "默认",
            "sandbox": "workspace-write",
            "approval_policy": "on-request",
            "approvals_reviewer": "",
        },
        "auto_review": {
            "label": "自动审批",
            "sandbox": "workspace-write",
            "approval_policy": "on-request",
            "approvals_reviewer": "auto_review",
        },
        "read_only": {
            "label": "只读",
            "sandbox": "read-only",
            "approval_policy": "on-request",
            "approvals_reviewer": "",
        },
        "full_access": {
            "label": "完全访问",
            "sandbox": "danger-full-access",
            "approval_policy": "never",
            "approvals_reviewer": "",
        },
    }


def normalize_permission_profile(value: str) -> str:
    raw = (value or "").strip().lower().replace("_", "-")
    aliases = {
        "": "default",
        "default": "default",
        "workspace": "default",
        "safe": "default",
        "on-request": "default",
        "ask": "default",
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
        "full": "full_access",
        "full-access": "full_access",
        "danger": "full_access",
        "danger-full-access": "full_access",
        "danger-no-sandbox": "full_access",
        "yolo": "full_access",
    }
    return aliases.get(raw, "")


def codex_app_server_config_overrides(cfg: dict[str, Any] | None = None) -> list[str]:
    codex_cfg = cfg if isinstance(cfg, dict) else load_codex_cfg()
    sandbox = normalize_sandbox_mode(str(codex_cfg.get("sandbox") or "workspace-write"))
    approval_policy = str(codex_cfg.get("approval_policy") or "on-request").strip() or "on-request"
    approvals_reviewer = str(codex_cfg.get("approvals_reviewer") or "").strip()
    overrides = [
        "-c",
        f'sandbox_mode="{sandbox}"',
        "-c",
        f'approval_policy="{approval_policy}"',
    ]
    if approvals_reviewer:
        overrides.extend(["-c", f'approvals_reviewer="{approvals_reviewer}"'])
    return overrides


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed) or parsed <= 0:
        return default
    return parsed


def codex_app_server_turn_options(cfg: dict[str, Any] | None = None) -> dict[str, float]:
    """Timeouts for a single app-server turn.

    Long-running remote Codex tasks commonly exceed the transport default. Keep
    the values configurable from Hermes while preserving the transport's
    post-tool quiet watchdog behavior.
    """
    codex_cfg = cfg if isinstance(cfg, dict) else load_codex_cfg()
    turn_timeout = codex_cfg.get("turn_timeout_seconds", codex_cfg.get("turn_timeout", 1800.0))
    post_tool_quiet_timeout = codex_cfg.get(
        "post_tool_quiet_timeout_seconds",
        codex_cfg.get("post_tool_quiet_timeout", 90.0),
    )
    active_tool_timeout = codex_cfg.get(
        "active_tool_timeout_seconds",
        codex_cfg.get("active_tool_timeout", 3600.0),
    )
    notification_poll_timeout = codex_cfg.get(
        "notification_poll_timeout_seconds",
        codex_cfg.get("notification_poll_timeout", 0.25),
    )
    return {
        "turn_timeout": _positive_float(turn_timeout, 1800.0),
        "post_tool_quiet_timeout": _positive_float(post_tool_quiet_timeout, 90.0),
        "active_tool_timeout": _positive_float(active_tool_timeout, 3600.0),
        "notification_poll_timeout": _positive_float(notification_poll_timeout, 0.25),
    }


def get_approval_callback():
    try:
        from tools.terminal_tool import _get_approval_callback

        return _get_approval_callback()
    except Exception:
        return None
