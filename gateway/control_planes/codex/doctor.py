"""Health checks for the platform-neutral Codex control plane."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
from typing import Any

from .event_store import CodexRuntimeEventStore
from .runtime_config import (
    codex_app_server_config_overrides,
    codex_app_server_turn_options,
    load_codex_cfg,
    normalize_sandbox_mode,
    read_codex_config_model,
)


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str = ""


def run_codex_doctor(
    *,
    workspace: str,
    task_key: str,
    selected_record: Any = None,
    event_store: CodexRuntimeEventStore | None = None,
    codex_bin: str = "codex",
) -> list[DoctorCheck]:
    cfg = load_codex_cfg()
    checks: list[DoctorCheck] = []
    binary = shutil.which(codex_bin)
    checks.append(
        DoctorCheck(
            "Codex CLI",
            "pass" if binary else "fail",
            binary or f"没有找到 {codex_bin}",
        )
    )
    checks.append(_check_codex_version(binary or codex_bin))
    checks.append(_check_app_server(binary or codex_bin))
    checks.append(_check_config(cfg))
    checks.append(_check_workspace(workspace))
    checks.append(_check_event_store(event_store or CodexRuntimeEventStore()))
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
    lines = ["Codex 诊断"]
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


def _check_codex_version(codex_bin: str) -> DoctorCheck:
    try:
        proc = subprocess.run(
            [codex_bin, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return DoctorCheck("Codex 版本", "warn", str(exc))
    output = (proc.stdout or proc.stderr or "").strip()
    return DoctorCheck(
        "Codex 版本",
        "pass" if proc.returncode == 0 else "warn",
        output or f"退出码 {proc.returncode}",
    )


def _check_app_server(codex_bin: str) -> DoctorCheck:
    try:
        proc = subprocess.run(
            [codex_bin, "app-server", "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return DoctorCheck("app-server", "fail", str(exc))
    output = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode != 0:
        return DoctorCheck("app-server", "fail", output or f"退出码 {proc.returncode}")
    return DoctorCheck("app-server", "pass", "可启动 help")


def _check_config(cfg: dict[str, Any]) -> DoctorCheck:
    sandbox = normalize_sandbox_mode(str(cfg.get("sandbox") or "workspace-write"))
    model = str(cfg.get("model") or read_codex_config_model())
    approval = str(cfg.get("approval_policy") or "on-request")
    turn_options = codex_app_server_turn_options(cfg)
    overrides = codex_app_server_config_overrides(cfg)
    detail = (
        f"model={model}; sandbox={sandbox}; approval={approval}; "
        f"turn_timeout={turn_options['turn_timeout']:.0f}s; overrides={len(overrides)//2}"
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


def _check_event_store(store: CodexRuntimeEventStore) -> DoctorCheck:
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
