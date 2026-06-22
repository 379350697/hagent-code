"""Human-friendly Chinese progress summaries for Claude runtime events."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import time
from typing import Any

from .event_store import ClaudeRuntimeEvent


@dataclass(frozen=True)
class ClaudeNarration:
    text: str
    importance: str = "normal"
    evidence: list[str] = field(default_factory=list)
    dedupe_key: str = ""

    @property
    def force(self) -> bool:
        return self.importance in {"high", "critical"}

    def render(self) -> str:
        lines = [self.text.strip()]
        for item in self.evidence[:2]:
            value = item.strip()
            if value:
                lines.append(f"证据：{value}")
        return "\n".join(lines)


class ClaudeFieldNarrator:
    """Project low-level Claude events into concise, human progress updates."""

    def narrate(
        self,
        event: ClaudeRuntimeEvent,
        *,
        recent_events: list[ClaudeRuntimeEvent] | None = None,
        workspace: str = "",
        thread_id: str = "",
    ) -> ClaudeNarration | None:
        event_type = event.event_type
        payload = event.payload
        workspace_name = _workspace_name(workspace or str(payload.get("cwd") or ""))
        short_thread = (thread_id or event.thread_id)[:13] or "pending"
        runtime_name = _runtime_name(payload.get("runtime") or payload.get("from"))

        if event_type in {"progress.session_ready", "usage.updated"}:
            return None
        if event_type == "progress.turn_started":
            return ClaudeNarration(
                "我开始接这轮任务了，会只在关键节点同步进展。",
                importance="high",
                evidence=[f"工作区 {workspace_name}，会话 {short_thread}"],
                dedupe_key="turn_started",
            )
        if event_type == "progress.approval_requested" or event_type == "approval.requested":
            command = _approval_command(payload)
            evidence = [f"待执行 {command}"] if command else ["Claude 正在等待权限决定"]
            return ClaudeNarration(
                "现在卡在审批，不是模型没反应；你点通过后我会继续。",
                importance="critical",
                evidence=evidence,
                dedupe_key="approval_requested",
            )
        if event_type == "progress.turn_timed_out":
            timeout = int(float(payload.get("timeout_seconds") or 0))
            return ClaudeNarration(
                f"这里像是 {runtime_name} 没继续吐事件，我会中断这轮并让下一轮干净恢复。",
                importance="critical",
                evidence=[f"最近活动超过 {timeout} 秒"] if timeout else [],
                dedupe_key="turn_timed_out",
            )
        if event_type == "turn.unconfirmed":
            error = _one_line(_localize_runtime_error(str(payload.get("error") or "未知")), 160)
            return ClaudeNarration(
                "Hermes 没确认到这轮的最终事件；我不会替 Claude 判失败，会让下一轮从干净状态恢复。",
                importance="critical",
                evidence=[error] if error else [],
                dedupe_key="turn_unconfirmed",
            )
        if event_type == "turn.failed":
            error = _one_line(_localize_runtime_error(str(payload.get("error") or "未知错误")), 160)
            if str(payload.get("error_kind") or "") == "api_retry_non_zero_exit":
                return ClaudeNarration(
                    "这轮 Claude 上游 API 连续重试后失败了，不是 Hermes 找不到 CLI。",
                    importance="critical",
                    evidence=[error] if error else [],
                    dedupe_key="turn_failed_api_retry",
                )
            return ClaudeNarration(
                f"这轮 {runtime_name} 断流了，我已中断本轮，下一轮会从干净状态继续。",
                importance="critical",
                evidence=[error] if error else [],
                dedupe_key="turn_failed",
            )
        if event_type == "runtime.fallback":
            target = _runtime_name(payload.get("to") or "cli")
            reason = _one_line(_localize_runtime_error(str(payload.get("reason") or "")), 160)
            return ClaudeNarration(
                f"Claude SDK 主链路不可用，已自动回退到 {target}。",
                importance="high",
                evidence=[reason] if reason else [],
                dedupe_key="runtime_fallback",
            )
        if event_type == "turn.completed" or event_type == "progress.turn_completed":
            if str(payload.get("error_kind") or "") == "success_result_then_exit_1":
                warning = _one_line(
                    _localize_runtime_error(str(payload.get("warning") or "")),
                    160,
                )
                return ClaudeNarration(
                    "Claude 已返回成功结果，但 CLI 最后退出码异常；我保留结果并记录提醒。",
                    importance="high",
                    evidence=[warning] if warning else [f"会话 {short_thread}"],
                    dedupe_key="turn_completed_with_warning",
                )
            return ClaudeNarration(
                "这轮 Claude 已经收尾，最终结果马上发出来。",
                importance="high",
                evidence=[f"会话 {short_thread}"],
                dedupe_key="turn_completed",
            )

        method = str(payload.get("method") or "")
        item = payload.get("item") if isinstance(payload.get("item"), dict) else {}

        if event_type == "claude.notification" and method == "thread/tokenUsage/updated":
            return None
        if method == "item/started" and item.get("type") == "tool_use":
            tool_name = str(item.get("name") or "unknown")
            return ClaudeNarration(
                "我正在调用工具验证现场，不是空等。",
                evidence=[tool_name] if tool_name else [],
                dedupe_key=f"tool_started:{tool_name}",
            )
        if method == "item/completed" and item.get("type") == "tool_use":
            tool_name = str(item.get("name") or "unknown")
            return ClaudeNarration(
                "刚完成一次工具调用，我会继续顺着结果往下查。",
                evidence=[tool_name] if tool_name else [],
                dedupe_key=f"tool_completed:{tool_name}",
            )
        if method == "item/completed" and item.get("type") == "agentMessage":
            # Final model text is delivered by the normal result path; avoid
            # echoing model prose as progress noise.
            return None
        if event_type == "progress.tool_completed":
            count = int(payload.get("tool_iterations") or 0)
            recent = _recent_activity_evidence(
                recent_events,
                current_event_id=event.id,
                limit=2,
            )
            return ClaudeNarration(
                "我刚完成一次操作，正在根据最近结果继续处理。",
                evidence=recent or ([f"已完成 {count} 次操作"] if count else []),
                dedupe_key=f"tool_completed:{count}",
            )
        return None

    def status_text(
        self,
        events: list[ClaudeRuntimeEvent],
        *,
        workspace: str = "",
        thread_id: str = "",
    ) -> str:
        terminal_priority = {
            "turn.completed",
            "progress.turn_completed",
            "turn.failed",
            "turn.unconfirmed",
            "task.recoverable_stale",
            "runtime.fallback",
        }
        for event in reversed(events):
            if event.event_type not in terminal_priority:
                continue
            narration = self.narrate(
                event,
                recent_events=events,
                workspace=workspace,
                thread_id=thread_id,
            )
            if narration is not None:
                age = _duration(time.time() - event.occurred_at)
                return f"{narration.render()}\n最后活动：{age}前"
        for event in reversed(events):
            narration = self.narrate(
                event,
                recent_events=events,
                workspace=workspace,
                thread_id=thread_id,
            )
            if narration is not None:
                age = _duration(time.time() - event.occurred_at)
                return f"{narration.render()}\n最后活动：{age}前"
        if events:
            age = _duration(time.time() - events[-1].occurred_at)
            return f"最近有 Claude 活动，但没有需要打扰你的关键节点。\n最后活动：{age}前"
        return ""

    def format_event_line(self, event: ClaudeRuntimeEvent) -> str:
        payload = event.payload
        method = str(payload.get("method") or "")
        suffix = f" · {method}" if method else ""
        age = _duration(time.time() - event.occurred_at)
        return f"#{event.id} {age}前 · {event.event_type}{suffix}"


def event_type_from_progress(progress: dict[str, Any]) -> str:
    stage = str(progress.get("stage") or "").strip()
    if stage == "notification":
        method = str(progress.get("method") or "")
        if method == "thread/tokenUsage/updated":
            return "usage.updated"
        return "claude.notification"
    if stage == "server_request":
        return "approval.requested"
    if stage == "turn_completed":
        return "progress.turn_completed"
    if stage == "error":
        return "turn.failed"
    if stage:
        return f"progress.{stage}"
    return "progress.unknown"


def _recent_activity_evidence(
    events: list[ClaudeRuntimeEvent] | None,
    *,
    current_event_id: int = 0,
    limit: int = 2,
) -> list[str]:
    if not events or limit <= 0:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for event in reversed(events):
        if current_event_id and event.id == current_event_id:
            continue
        summary = _activity_summary(event)
        summary = _one_line(summary, 160)
        if not summary or summary in seen:
            continue
        seen.add(summary)
        result.append(summary)
        if len(result) >= limit:
            break
    return result


def _activity_summary(event: ClaudeRuntimeEvent) -> str:
    payload = event.payload
    event_type = event.event_type
    if payload.get("truncated"):
        return ""
    if event_type in {"progress.approval_requested", "approval.requested"}:
        command = _approval_command(payload)
        return f"等待审批：{command}" if command else "等待审批"
    if event_type == "progress.tool_completed":
        count = int(_float_or_zero(payload.get("tool_iterations")))
        return f"已完成 {count} 次操作" if count else ""

    method = str(payload.get("method") or "")
    item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
    item_type = str(item.get("type") or "")
    if method == "item/started" and item_type == "tool_use":
        tool = _one_line(str(item.get("name") or item_type), 80)
        return f"正在调用工具：{tool}" if tool else "正在调用工具"
    if method == "item/completed" and item_type == "tool_use":
        tool = _one_line(str(item.get("name") or item_type), 80)
        return f"工具调用完成：{tool}" if tool else "工具调用完成"
    return ""


def _approval_command(payload: dict[str, Any]) -> str:
    command = payload.get("command") or ""
    if not command:
        request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        command = params.get("command") or ""
    return _one_line(str(command), 120)


def _workspace_name(workspace: str) -> str:
    value = str(workspace or "").rstrip(os.sep)
    return os.path.basename(value) if value else "(unknown)"


def _one_line(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) > limit:
        return value[: limit - 3] + "..."
    return value


def _float_or_zero(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _runtime_name(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"agent_sdk", "agent-sdk", "sdk"}:
        return "Claude SDK"
    if raw in {"cli", "claude_cli", "claude-cli", "claude_code", "claude-code"}:
        return "Claude CLI"
    return "Claude runtime"


def _localize_runtime_error(text: str) -> str:
    value = re.sub(
        r"claude turn timed out after ([0-9.]+)s \(([^)]+)\)",
        r"Claude 本轮在 \1 秒后超时（\2）",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    value = value.replace(
        "claude binary not found",
        "Claude CLI 未安装或路径错误",
    )
    return value


def _duration(seconds: float) -> str:
    value = max(0, int(seconds))
    if value < 60:
        return f"{value} 秒"
    minutes = value // 60
    if minutes < 60:
        return f"{minutes} 分钟"
    return f"{minutes // 60} 小时"
