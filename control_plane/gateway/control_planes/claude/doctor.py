"""Health checks for the platform-neutral Claude control plane."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import subprocess
from typing import Any

from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession
from agent.transports.claude_cli_session import resolve_claude_binary

from .event_store import ClaudeRuntimeEventStore
from .runtime_config import (
    claude_runtime,
    claude_runtime_fallback,
    claude_cli_turn_options,
    load_claude_cfg,
    normalize_permission_mode,
    read_claude_config_model,
    resolve_claude_sdk_profile,
    safe_claude_sdk_profile_diagnostics,
)


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str = ""


def run_claude_doctor(
    *,
    workspace: str,
    task_key: str,
    selected_record: Any = None,
    event_store: ClaudeRuntimeEventStore | None = None,
    claude_bin: str = "",
) -> list[DoctorCheck]:
    cfg = load_claude_cfg()
    checks: list[DoctorCheck] = []
    runtime = claude_runtime(cfg)
    sdk_profile = resolve_claude_sdk_profile(cfg)
    checks.append(_check_sdk_runtime(runtime, sdk_profile))
    binary = claude_bin or resolve_claude_binary(str(cfg.get("binary") or "auto")) or ""
    checks.append(
        DoctorCheck(
            "Claude CLI",
            "pass" if binary else "fail",
            binary or "没有找到 claude；可用 HERMES_CLAUDE_BINARY 指定",
        )
    )
    checks.append(_check_claude_version(binary or "claude"))
    checks.append(_check_config(cfg))
    if runtime == "agent_sdk":
        checks.append(_check_sdk_smoke(workspace, sdk_profile, cfg))
    elif binary:
        checks.append(_check_claude_smoke(binary, workspace))
    checks.append(_check_workspace(workspace))
    checks.append(_check_event_store(event_store or ClaudeRuntimeEventStore()))
    checks.append(_check_selected_session(selected_record))
    checks.append(
        DoctorCheck(
            "会话隔离键",
            "pass" if task_key else "fail",
            task_key or "缺少 platform/chat/thread key",
        )
    )
    return checks


def format_doctor_checks(checks: list[DoctorCheck]) -> str:
    icons = {"pass": "通过", "warn": "提醒", "fail": "失败"}
    lines = ["Claude 诊断"]

    for line in _format_user_facing_checks(checks, icons):
        lines.append(line)

    for check in checks:
        if _hide_successful_internal_check(check):
            continue
        if _is_user_facing_check(check):
            continue
        label = icons.get(check.status, check.status)
        suffix = f"：{check.detail}" if check.detail else ""
        lines.append(f"- {label} · {check.name}{suffix}")
    failed = sum(1 for item in checks if item.status == "fail")
    warned = sum(1 for item in checks if item.status == "warn")
    if failed:
        lines.append(f"结论：有 {failed} 项失败，先修失败项再跑长任务。")
    elif warned:
        lines.append(f"结论：核心链路可用，有 {warned} 项提醒。")
    else:
        lines.append("结论：核心链路可用。")
    return "\n".join(lines)


def _format_user_facing_checks(
    checks: list[DoctorCheck],
    icons: dict[str, str],
) -> list[str]:
    by_name = {check.name: check for check in checks}
    lines: list[str] = []

    sdk = by_name.get("Claude SDK")
    config = by_name.get("运行配置")
    cli = by_name.get("Claude CLI")
    if sdk is not None:
        detail = _friendly_sdk_detail(sdk, config, cli)
        lines.append(_format_line("Claude 可用性", sdk.status, detail, icons))

    if config is not None:
        lines.append(
            _format_line(
                "运行模式",
                config.status,
                _friendly_config_detail(config.detail),
                icons,
            )
        )

    smoke = by_name.get("Claude SDK smoke test") or by_name.get("Claude smoke test")
    if smoke is not None:
        lines.append(
            _format_line(
                "短任务测试",
                smoke.status,
                _friendly_smoke_detail(smoke.detail),
                icons,
            )
        )

    workspace = by_name.get("工作区")
    if workspace is not None:
        lines.append(_format_line("工作区", workspace.status, workspace.detail, icons))

    session = by_name.get("当前会话")
    if session is not None:
        lines.append(_format_line("当前会话", session.status, session.detail, icons))

    return lines


def _format_line(name: str, status: str, detail: str, icons: dict[str, str]) -> str:
    suffix = f"：{detail}" if detail else ""
    return f"- {icons.get(status, status)} · {name}{suffix}"


def _is_user_facing_check(check: DoctorCheck) -> bool:
    return check.name in {
        "Claude SDK",
        "运行配置",
        "Claude SDK smoke test",
        "Claude smoke test",
        "工作区",
        "当前会话",
    }


def _hide_successful_internal_check(check: DoctorCheck) -> bool:
    return check.status == "pass" and check.name in {
        "Claude CLI",
        "Claude 版本",
        "事件库",
        "会话隔离键",
    }


def _friendly_sdk_detail(
    sdk: DoctorCheck,
    config: DoctorCheck | None,
    cli: DoctorCheck | None,
) -> str:
    if sdk.status != "pass":
        return sdk.detail
    profile = _detail_value(sdk.detail, "profile") or _detail_value(
        config.detail if config else "",
        "profile",
    )
    model = _detail_value(sdk.detail, "model") or _detail_value(
        config.detail if config else "",
        "model",
    )
    parts = ["SDK 已连接"]
    profile_model = " / ".join(part for part in [profile, model] if part)
    if profile_model:
        parts.append(f"（{profile_model}）")
    if cli is not None and cli.status == "pass":
        parts.append("；CLI 备选可用")
    return "".join(parts)


def _friendly_config_detail(detail: str) -> str:
    runtime = _detail_value(detail, "runtime")
    fallback = _detail_value(detail, "fallback")
    permission = _detail_value(detail, "permission_mode")
    turn_timeout = _detail_value(detail, "turn_timeout")
    runtime_label = "SDK 默认" if runtime == "agent_sdk" else (runtime or "默认")
    if fallback and fallback != "none":
        runtime_label = f"{runtime_label}，{fallback.upper()} 备选"
    parts = [runtime_label]
    if permission:
        parts.append(f"权限={permission}")
    if turn_timeout:
        parts.append(f"最长任务={turn_timeout}")
    return "；".join(parts)


def _friendly_smoke_detail(detail: str) -> str:
    return detail or "OK"


def _detail_value(detail: str, key: str) -> str:
    prefix = f"{key}="
    for part in detail.split(";"):
        item = part.strip()
        if item.startswith(prefix):
            return item[len(prefix) :].strip()
    return ""


def _check_claude_version(claude_bin: str) -> DoctorCheck:
    try:
        proc = subprocess.run(
            [claude_bin, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return DoctorCheck("Claude 版本", "warn", str(exc))
    output = (proc.stdout or proc.stderr or "").strip()
    return DoctorCheck(
        "Claude 版本",
        "pass" if proc.returncode == 0 else "warn",
        output or f"退出码 {proc.returncode}",
    )


def _check_config(cfg: dict[str, Any]) -> DoctorCheck:
    runtime = claude_runtime(cfg)
    fallback = claude_runtime_fallback(cfg) or "none"
    sdk_profile = resolve_claude_sdk_profile(cfg)
    sdk_diag = safe_claude_sdk_profile_diagnostics(sdk_profile)
    permission_mode = normalize_permission_mode(
        str(cfg.get("permission_mode") or "acceptEdits")
    )
    model = str(cfg.get("model") or sdk_profile.get("model") or read_claude_config_model())
    turn_options = claude_cli_turn_options(cfg)
    turn_timeout = turn_options["turn_timeout"]
    idle_timeout = turn_options["idle_timeout"]
    detail = (
        f"runtime={runtime}; fallback={fallback}; "
        f"profile={sdk_diag['name']}; model={model}; permission_mode={permission_mode}; "
        f"turn_timeout={turn_timeout:.0f}s; "
        f"idle_timeout={idle_timeout:.0f}s"
    )
    return DoctorCheck("运行配置", "pass", detail)


def _check_sdk_runtime(runtime: str, profile: dict[str, Any]) -> DoctorCheck:
    diagnostics = safe_claude_sdk_profile_diagnostics(profile)
    spec = importlib.util.find_spec("claude_agent_sdk")
    details = (
        f"profile={diagnostics['name']}; base_url={diagnostics['base_url'] or 'anthropic-default'}; "
        f"model={diagnostics['model']}; key_source={diagnostics['api_key_source'] or 'none'}"
    )
    if spec is None:
        status = "fail" if runtime == "agent_sdk" else "warn"
        return DoctorCheck("Claude SDK", status, f"claude-agent-sdk 未安装；{details}")
    if not diagnostics["api_key_available"]:
        status = "fail" if runtime == "agent_sdk" else "warn"
        return DoctorCheck("Claude SDK", status, f"缺少 API key；{details}")
    return DoctorCheck("Claude SDK", "pass", details)


def _check_sdk_smoke(workspace: str, profile: dict[str, Any], cfg: dict[str, Any]) -> DoctorCheck:
    enabled = os.environ.get("HERMES_CLAUDE_DOCTOR_SMOKE", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return DoctorCheck("Claude SDK smoke test", "warn", "已跳过；设置 HERMES_CLAUDE_DOCTOR_SMOKE=1 可启用")
    try:
        session = ClaudeAgentSdkSession(
            cwd=workspace or os.getcwd(),
            permission_mode="plan",
            model=str(cfg.get("model") or profile.get("model") or ""),
            effort=str(cfg.get("effort") or profile.get("effort") or ""),
            sdk_profile_config=profile,
        )
        result = session.run_turn(
            "Reply with OK only.",
            turn_timeout=45.0,
            idle_timeout=30.0,
        )
    except Exception as exc:
        return DoctorCheck("Claude SDK smoke test", "warn", str(exc))
    if result.error:
        return DoctorCheck("Claude SDK smoke test", "warn", result.error[:240])
    return DoctorCheck("Claude SDK smoke test", "pass", (result.final_text or "OK")[:200])


def _check_claude_smoke(claude_bin: str, workspace: str) -> DoctorCheck:
    enabled = os.environ.get("HERMES_CLAUDE_DOCTOR_SMOKE", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return DoctorCheck("Claude smoke test", "warn", "已跳过；设置 HERMES_CLAUDE_DOCTOR_SMOKE=1 可启用")
    cwd = workspace if workspace and os.path.isdir(workspace) else None
    try:
        proc = subprocess.run(
            [
                claude_bin,
                "-p",
                "Reply with OK only.",
                "--output-format",
                "json",
                "--permission-mode",
                "plan",
            ],
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=25,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return DoctorCheck("Claude smoke test", "warn", "25s 内没有完成；可能是登录、网络或上游 API 问题")
    except Exception as exc:
        return DoctorCheck("Claude smoke test", "warn", str(exc))
    output = " ".join((proc.stdout or proc.stderr or "").split())
    if proc.returncode == 0:
        return DoctorCheck("Claude smoke test", "pass", output[:200] or "OK")
    return DoctorCheck(
        "Claude smoke test",
        "warn",
        f"退出码 {proc.returncode}; {output[:240] or '无输出'}",
    )


def _check_workspace(workspace: str) -> DoctorCheck:
    if not workspace:
        return DoctorCheck("工作区", "warn", "未指定，使用当前目录")
    if not os.path.isdir(workspace):
        return DoctorCheck("工作区", "fail", f"不存在：{workspace}")
    if os.path.isdir(os.path.join(workspace, ".git")):
        return DoctorCheck("工作区", "pass", workspace)
    return DoctorCheck("工作区", "warn", f"不是 git 根目录：{workspace}")


def _check_event_store(store: ClaudeRuntimeEventStore) -> DoctorCheck:
    try:
        events = store.tail(limit=1)
    except Exception as exc:
        return DoctorCheck("事件库", "fail", str(exc))
    detail = f"{store.path}; 最近事件 {len(events)} 条"
    return DoctorCheck("事件库", "pass", detail)


def _check_selected_session(record: Any) -> DoctorCheck:
    if record is None:
        return DoctorCheck("当前会话", "warn", "当前聊天还没有选中会话")
    thread_id = str(getattr(record, "thread_id", "") or "")
    workspace = str(getattr(record, "workspace", "") or "")
    if not thread_id:
        return DoctorCheck("当前会话", "warn", "记录存在但没有 thread_id")
    return DoctorCheck("当前会话", "pass", f"{thread_id[:8]} · {workspace}")
