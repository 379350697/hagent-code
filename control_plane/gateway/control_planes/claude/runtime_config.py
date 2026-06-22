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
DEFAULT_CLAUDE_RUNTIME = "agent_sdk"
DEFAULT_CLAUDE_RUNTIME_FALLBACK = "cli"
DEFAULT_CLAUDE_SDK_PROFILE = "opencodego"


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


def read_claude_settings_env() -> dict[str, str]:
    """Read Claude settings env without exposing values to callers."""
    try:
        import json

        path = os.path.expanduser("~/.claude/settings.json")
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        candidates = data.get("env") if isinstance(data, dict) else {}
        if not isinstance(candidates, dict):
            candidates = data if isinstance(data, dict) else {}
        result: dict[str, str] = {}
        for key, value in candidates.items():
            if isinstance(value, str):
                result[str(key)] = value
        return result
    except Exception:
        return {}


def load_claude_cfg() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        value = cfg.get("claude_cli", {}) if isinstance(cfg, dict) else {}
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def claude_runtime(cfg: dict[str, Any] | None = None) -> str:
    """Configured Claude runtime.

    Defaults to the SDK runtime; CLI remains the configured fallback.
    """
    claude_cfg = cfg if isinstance(cfg, dict) else load_claude_cfg()
    raw = str(
        claude_cfg.get("runtime")
        or os.environ.get("HERMES_CLAUDE_RUNTIME")
        or DEFAULT_CLAUDE_RUNTIME
    ).strip().lower()
    aliases = {
        "": DEFAULT_CLAUDE_RUNTIME,
        "cli": "cli",
        "claude_cli": "cli",
        "claude-code": "cli",
        "claude_code": "cli",
        "agent-sdk": "agent_sdk",
        "agent_sdk": "agent_sdk",
        "sdk": "agent_sdk",
    }
    return aliases.get(raw, raw)


def claude_runtime_fallback(cfg: dict[str, Any] | None = None) -> str:
    claude_cfg = cfg if isinstance(cfg, dict) else load_claude_cfg()
    raw = str(
        claude_cfg.get("runtime_fallback")
        or os.environ.get("HERMES_CLAUDE_RUNTIME_FALLBACK")
        or DEFAULT_CLAUDE_RUNTIME_FALLBACK
    ).strip().lower()
    aliases = {
        "": "",
        "none": "",
        "off": "",
        "cli": "cli",
        "claude_cli": "cli",
        "agent-sdk": "agent_sdk",
        "agent_sdk": "agent_sdk",
        "sdk": "agent_sdk",
    }
    return aliases.get(raw, raw)


def claude_sdk_profile_name(cfg: dict[str, Any] | None = None) -> str:
    claude_cfg = cfg if isinstance(cfg, dict) else load_claude_cfg()
    raw = str(
        os.environ.get("HERMES_CLAUDE_SDK_PROFILE")
        or claude_cfg.get("sdk_profile")
        or DEFAULT_CLAUDE_SDK_PROFILE
    ).strip().lower()
    aliases = {
        "": DEFAULT_CLAUDE_SDK_PROFILE,
        "open-code-go": "opencodego",
        "open_code_go": "opencodego",
        "opencode-go": "opencodego",
        "opencode_go": "opencodego",
    }
    return aliases.get(raw, raw)


def resolve_claude_sdk_profile(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    claude_cfg = cfg if isinstance(cfg, dict) else load_claude_cfg()
    name = claude_sdk_profile_name(claude_cfg)
    configured_profiles = claude_cfg.get("sdk_profiles")
    configured = (
        configured_profiles.get(name, {})
        if isinstance(configured_profiles, dict)
        and isinstance(configured_profiles.get(name, {}), dict)
        else {}
    )
    settings_env = read_claude_settings_env()
    profile = _default_sdk_profile(name, settings_env)
    profile.update({k: v for k, v in configured.items() if v not in (None, "")})

    env_base_url = os.environ.get("HERMES_CLAUDE_SDK_BASE_URL", "").strip()
    env_api_key_env = os.environ.get("HERMES_CLAUDE_SDK_API_KEY_ENV", "").strip()
    env_model = os.environ.get("HERMES_CLAUDE_SDK_MODEL", "").strip()
    if env_base_url:
        profile["base_url"] = env_base_url
    if env_api_key_env:
        profile["api_key_env"] = env_api_key_env
        profile.pop("api_key", None)
        profile["api_key_source"] = f"env:{env_api_key_env}"
    if env_model:
        profile["model"] = env_model
    cli_path = str(os.environ.get("HERMES_CLAUDE_BINARY") or claude_cfg.get("binary") or "").strip()
    if cli_path and cli_path.lower() != "auto":
        profile["cli_path"] = cli_path

    if name == "opencodego":
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            if settings_env.get(key) and not profile.get("api_key"):
                profile["api_key"] = settings_env[key]
                profile["api_key_source"] = "~/.claude/settings.json"
        if settings_env.get("ANTHROPIC_BASE_URL") and not env_base_url:
            profile["base_url"] = settings_env["ANTHROPIC_BASE_URL"]

    profile["name"] = name
    profile.setdefault("api_key_source", profile.get("api_key_env", ""))
    profile.setdefault("extra_env", {})
    return profile


def safe_claude_sdk_profile_diagnostics(profile: dict[str, Any]) -> dict[str, Any]:
    api_key_env = str(profile.get("api_key_env") or "")
    key_source = str(profile.get("api_key_source") or "")
    key_available = bool(profile.get("api_key") or (api_key_env and os.environ.get(api_key_env)))
    return {
        "name": str(profile.get("name") or ""),
        "base_url": str(profile.get("base_url") or ""),
        "model": str(profile.get("model") or ""),
        "effort": str(profile.get("effort") or ""),
        "api_key_env": api_key_env,
        "api_key_source": key_source,
        "api_key_available": key_available,
    }


def _default_sdk_profile(name: str, settings_env: dict[str, str]) -> dict[str, Any]:
    if name == "deepseek":
        return {
            "name": "deepseek",
            "base_url": os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/anthropic",
            "api_key_env": "DEEPSEEK_API_KEY",
            "api_key_source": "env:DEEPSEEK_API_KEY",
            "model": "deepseek-v4-pro",
            "effort": DEFAULT_CLAUDE_EFFORT,
        }
    if name == "anthropic":
        return {
            "name": "anthropic",
            "base_url": "",
            "api_key_env": "ANTHROPIC_API_KEY",
            "api_key_source": "env:ANTHROPIC_API_KEY",
            "model": DEFAULT_CLAUDE_MODEL,
            "effort": DEFAULT_CLAUDE_EFFORT,
        }
    return {
        "name": "opencodego",
        "base_url": settings_env.get("ANTHROPIC_BASE_URL") or "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODEGO_API_KEY",
        "api_key_source": "env:OPENCODEGO_API_KEY",
        "model": settings_env.get("ANTHROPIC_MODEL") or "deepseek-v4-pro",
        "effort": DEFAULT_CLAUDE_EFFORT,
    }


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
