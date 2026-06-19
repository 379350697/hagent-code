"""User-facing formatting for Codex command results."""

from __future__ import annotations

import re
from typing import Any


def format_failure(prefix: str, error: Any) -> str:
    message = _localize_message(str(error or "未知错误"))
    lowered = message.lower()
    if "codex approval bridge unavailable" in lowered or "codex 审批通道不可用" in message:
        return "Codex 审批通道不可用：当前聊天没有接上审批回调。"
    if (
        "codex approval denied" in lowered
        or "codex approval timed out" in lowered
        or "codex 审批已拒绝" in message
        or "codex 审批已超时" in message
    ):
        return "Codex 审批未通过或已超时。"
    if "bwrap" in lowered or "bubblewrap" in lowered or "sandbox" in lowered:
        return f"Codex 沙箱启动失败：{message}"
    if "timed out" in lowered or "timeout" in lowered:
        return f"Codex 任务超时：{message}"
    return f"{_localize_prefix(prefix)}：{message}"


def format_task_status(record: Any) -> str:
    lines = [
        f"Codex 会话：{getattr(record, 'task_id', '')}",
        f"最近一轮：{_status_label(getattr(record, 'status', '') or '')}",
        f"线程：{getattr(record, 'thread_id', '') or '未知'}",
    ]
    turn_id = getattr(record, "turn_id", "")
    if turn_id:
        lines.append(f"轮次：{turn_id}")
    workspace = getattr(record, "workspace", "")
    if workspace:
        lines.append(f"工作区：{workspace}")
    lines.append(
        "权限："
        f"{_sandbox_label(getattr(record, 'sandbox', 'workspace-write'))} / "
        f"{_approval_label(getattr(record, 'approval_policy', 'on-request'))}"
    )
    model = getattr(record, "model", "")
    if model:
        lines.append(f"模型：{model}")
    title = _display_title(getattr(record, "title", ""))
    if title:
        lines.append(f"任务：{title}")
    last = getattr(record, "last_message", "")
    if last:
        lines.append(f"最近状态：{_localize_message(last)}")
    return "\n".join(lines)


def format_run_success(workspace: str, thread_id: str, message: str) -> str:
    body = message.strip() or "Codex 已完成，但没有返回最终文本。"
    return (
        "Codex 结果\n"
        f"工作区：{workspace}\n"
        f"会话：{thread_id}\n\n"
        f"{body}"
    )


def format_run_failure(workspace: str, thread_id: str, error: str) -> str:
    return (
        "Codex 任务失败\n"
        f"工作区：{workspace}\n"
        f"会话：{thread_id or '新会话'}\n"
        f"错误：{format_failure('Codex app-server failed', error)}"
    )


def _localize_prefix(prefix: str) -> str:
    mapping = {
        "Codex app-server failed": "Codex app-server 失败",
        "Codex sandbox failed": "Codex 沙箱启动失败",
    }
    return mapping.get(prefix, prefix)


def _localize_message(message: str) -> str:
    replacements = {
        "Codex: turn started": "Codex 已开始处理",
        "Codex: turn completed": "Codex 已完成本轮",
        "Codex: turn interrupted": "Codex 本轮已中断",
        "Codex stop requested.": "已请求停止 Codex。",
        "unknown error": "未知错误",
        "turn timed out": "本轮超时",
        "without app-server activity": "没有收到 app-server 活动",
        "codex app-server subprocess exited unexpectedly": "Codex app-server 子进程意外退出",
    }
    text = message
    text = re.sub(
        r"codex went silent for ([0-9.]+)s after a tool result; retiring app-server session\.",
        r"Codex app-server 在工具步骤后 \1 秒没有新事件；已回收本轮运行时。",
        text,
        flags=re.IGNORECASE,
    )
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _status_label(status: str) -> str:
    return {
        "starting": "启动中",
        "planning": "规划中",
        "running": "运行中",
        "completed": "已完成",
        "failed": "失败",
        "interrupted": "已中断",
        "busy": "忙碌",
        "not_found": "未找到",
        "unknown": "未知",
    }.get(str(status), str(status))


def _sandbox_label(value: Any) -> str:
    raw = str(value or "")
    return {
        "workspace-write": "工作区可写",
        "read-only": "只读",
        "danger-full-access": "完全访问",
    }.get(raw, raw)


def _approval_label(value: Any) -> str:
    raw = str(value or "")
    return {
        "on-request": "按需审批",
        "never": "无需审批",
        "untrusted": "严格审批",
    }.get(raw, raw)


def _display_title(title: Any) -> str:
    value = " ".join(str(title or "").split())
    if value.startswith("Create a detailed implementation plan first."):
        return "计划会话"
    return value
