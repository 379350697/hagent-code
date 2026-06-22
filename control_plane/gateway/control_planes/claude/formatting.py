"""User-facing formatting for Claude command results."""

from __future__ import annotations

import re
from typing import Any


def format_failure(prefix: str, error: Any) -> str:
    message = _localize_message(str(error or "未知错误"))
    lowered = message.lower()
    if "claude approval bridge unavailable" in lowered or "claude 审批通道不可用" in message:
        return "Claude 审批通道不可用：当前聊天没有接上审批回调。"
    if (
        "claude approval denied" in lowered
        or "claude approval timed out" in lowered
        or "claude 审批已拒绝" in message
        or "claude 审批已超时" in message
    ):
        return "Claude 审批未通过或已超时。"
    if "binary not found" in lowered or "claude binary not found" in lowered:
        return (
            f"Claude CLI 未安装或路径错误：{message}\n"
            "用 `HERMES_CLAUDE_BINARY` 指定路径，或安装："
            "`npm install -g @anthropic-ai/claude-code`。"
        )
    if "timed out" in lowered or "timeout" in lowered or "超时" in message:
        return f"Claude 任务超时：{message}"
    if "api retries were exhausted" in lowered:
        return f"Claude 上游 API 连续重试后失败：{message}"
    if "claude-agent-sdk is not installed" in lowered or "sdk_dependency_missing" in lowered:
        return f"Claude SDK 依赖缺失：{message}"
    if "has no api key" in lowered or "sdk_auth_error" in lowered:
        return f"Claude SDK 认证配置不完整：{message}"
    if "sdk_rate_limit" in lowered or "rate limited" in lowered:
        return f"Claude SDK 上游限流：{message}"
    if "sdk_connection" in lowered:
        return f"Claude SDK 连接失败：{message}"
    if "successful result event" in lowered and "exited with status" in lowered:
        return f"Claude 已返回成功结果，但 CLI 退出码异常：{message}"
    if "exited with status" in lowered:
        return f"Claude CLI 退出码异常：{message}"
    if "permission" in lowered and "mode" in lowered:
        return f"Claude 权限模式问题：{message}"
    return f"{_localize_prefix(prefix)}：{message}"


def format_task_status(record: Any, *, verbose: bool = False) -> str:
    lines = [
        "Claude 状态",
        f"最近一轮：{_status_label(getattr(record, 'status', '') or '')}",
    ]
    if verbose:
        lines.append(f"任务 ID：{getattr(record, 'task_id', '')}")
        lines.append(f"线程：{getattr(record, 'thread_id', '') or '未知'}")
    turn_id = getattr(record, "turn_id", "")
    if turn_id:
        lines.append(f"轮次：{turn_id}")
    workspace = getattr(record, "workspace", "")
    if workspace:
        lines.append(f"工作区：{workspace}")
    lines.append(
        f"权限模式：{_permission_mode_label(getattr(record, 'permission_mode', 'acceptEdits'))}"
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
    del thread_id
    body = message.strip() or "Claude 已完成，但没有返回最终文本。"
    return (
        "Claude 结果\n"
        f"工作区：{workspace}\n"
        "会话：当前会话\n\n"
        f"{body}"
    )


def format_run_failure(workspace: str, thread_id: str, error: str) -> str:
    del thread_id
    return (
        "Claude 任务失败\n"
        f"工作区：{workspace}\n"
        "会话：当前会话\n"
        f"错误：{format_failure('Claude CLI failed', error)}"
    )


def _localize_prefix(prefix: str) -> str:
    mapping = {
        "Claude CLI failed": "Claude CLI 失败",
        "Claude CLI": "Claude CLI",
    }
    return mapping.get(prefix, prefix)


def _localize_message(message: str) -> str:
    replacements = {
        "Claude: turn started": "Claude 已开始处理",
        "Claude: turn completed": "Claude 已完成本轮",
        "Claude: turn interrupted": "Claude 本轮已中断",
        "Claude stop requested.": "已请求停止 Claude。",
        "Claude API retries were exhausted before the CLI exited": "Claude 上游 API 连续重试后 CLI 退出",
        "Claude CLI exited with status": "Claude CLI 退出码",
        "Claude returned a successful result event, but the CLI process exited": "Claude 已返回成功结果，但 CLI 进程退出码异常",
        "Claude SDK unavailable; fell back to cli.": "Claude SDK 不可用，已回退到 CLI。",
        "claude-agent-sdk is not installed": "claude-agent-sdk 未安装",
        "unknown error": "未知错误",
        "turn timed out": "本轮超时",
        "binary not found": "未找到 Claude CLI",
        "Claude binary not found": "未找到 Claude CLI",
    }
    text = message
    text = re.sub(
        r"claude turn timed out after ([0-9.]+)s \(([^)]+)\)",
        r"Claude 本轮在 \1 秒后超时（\2）",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"claude turn idle-timed out after ([0-9.]+)s.*",
        r"Claude 本轮在 \1 秒无活动后超时。",
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
        "unconfirmed": "未确认",
        "recoverable_stale": "待恢复",
        "busy": "忙碌",
        "not_found": "未找到",
        "unknown": "未知",
    }.get(str(status), str(status))


def _permission_mode_label(value: Any) -> str:
    raw = str(value or "")
    return {
        "acceptEdits": "允许编辑",
        "auto": "自动模式",
        "plan": "只规划",
        "default": "默认确认",
        "dontAsk": "不询问",
        "bypassPermissions": "跳过权限检查",
    }.get(raw, raw)


def _display_title(title: Any) -> str:
    value = " ".join(str(title or "").split())
    return value
