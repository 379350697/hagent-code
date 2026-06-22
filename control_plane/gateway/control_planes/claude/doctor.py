"""Health checks for the platform-neutral Claude control plane."""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
from typing import Any

from agent.transports.claude_cli_session import resolve_claude_binary

from .event_store import ClaudeRuntimeEventStore
from .runtime_config import (
    claude_cli_turn_options,
    load_claude_cfg,
    normalize_permission_mode,
    read_claude_config_model,
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
    for check in checks:
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
    permission_mode = normalize_permission_mode(
        str(cfg.get("permission_mode") or "acceptEdits")
    )
    model = str(cfg.get("model") or read_claude_config_model())
    turn_options = claude_cli_turn_options(cfg)
    turn_timeout = turn_options["turn_timeout"]
    idle_timeout = turn_options["idle_timeout"]
    detail = (
        f"model={model}; permission_mode={permission_mode}; "
        f"turn_timeout={turn_timeout:.0f}s; "
        f"idle_timeout={idle_timeout:.0f}s"
    )
    return DoctorCheck("运行配置", "pass", detail)


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
