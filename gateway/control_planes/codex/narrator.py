"""Human-friendly Chinese progress summaries for Codex runtime events."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import time
from typing import Any

from .event_store import CodexRuntimeEvent


@dataclass(frozen=True)
class CodexNarration:
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


class CodexFieldNarrator:
    """Project low-level Codex events into concise, human progress updates."""

    def narrate(
        self,
        event: CodexRuntimeEvent,
        *,
        recent_events: list[CodexRuntimeEvent] | None = None,
        workspace: str = "",
        thread_id: str = "",
    ) -> CodexNarration | None:
        event_type = event.event_type
        payload = event.payload
        workspace_name = _workspace_name(workspace or str(payload.get("cwd") or ""))
        short_thread = (thread_id or event.thread_id)[:13] or "pending"

        if event_type in {"progress.session_ready", "progress.waiting", "usage.updated"}:
            return None
        if event_type == "progress.turn_started":
            return CodexNarration(
                "我开始接这轮任务了，会只在关键节点同步进展。",
                importance="high",
                evidence=[f"工作区 {workspace_name}，会话 {short_thread}"],
                dedupe_key="turn_started",
            )
        if event_type == "progress.approval_requested" or event_type == "approval.requested":
            command = _approval_command(payload)
            evidence = [f"待执行 {command}"] if command else ["Codex 正在等待权限决定"]
            return CodexNarration(
                "现在卡在审批，不是模型没反应；你点通过后我会继续。",
                importance="critical",
                evidence=evidence,
                dedupe_key="approval_requested",
            )
        if event_type == "progress.turn_timed_out":
            timeout = int(float(payload.get("timeout_seconds") or 0))
            return CodexNarration(
                "这里像是 app-server 没继续吐事件，我会中断这轮并让下一轮干净恢复。",
                importance="critical",
                evidence=[f"最近活动超过 {timeout} 秒"] if timeout else [],
                dedupe_key="turn_timed_out",
            )
        if event_type == "turn.unconfirmed":
            error = _one_line(_localize_runtime_error(str(payload.get("error") or "未知")), 160)
            return CodexNarration(
                "Hermes 没确认到这轮的最终事件；我不会替 Codex 判失败，会让下一轮从干净状态恢复。",
                importance="critical",
                evidence=[error] if error else [],
                dedupe_key="turn_unconfirmed",
            )
        if event_type == "turn.failed":
            error = _one_line(_localize_runtime_error(str(payload.get("error") or "未知错误")), 160)
            return CodexNarration(
                "这轮 app-server 断流了，我已中断本轮，下一轮会从干净状态继续。",
                importance="critical",
                evidence=[error] if error else [],
                dedupe_key="turn_failed",
            )
        if event_type == "turn.completed" or event_type == "progress.turn_completed":
            return CodexNarration(
                "这轮 Codex 已经收尾，最终结果马上发出来。",
                importance="high",
                evidence=[f"会话 {short_thread}"],
                dedupe_key="turn_completed",
            )

        method = str(payload.get("method") or "")
        notification = payload.get("notification") if isinstance(payload.get("notification"), dict) else {}
        item = _event_item(notification)
        item_type = str(item.get("type") or "")

        if event_type == "codex.notification" and method == "thread/tokenUsage/updated":
            return None
        if method == "item/started" and item_type == "commandExecution":
            command = _command_preview(item)
            return CodexNarration(
                "我正在跑命令验证现场，不是空等。",
                evidence=[command] if command else [],
                dedupe_key=f"command_started:{_command_key(command)}",
            )
        if method == "item/completed" and item_type == "commandExecution":
            command = _command_preview(item)
            output = str(item.get("aggregatedOutput") or "")
            if _looks_like_test_command(command) or _looks_like_test_output(output):
                return CodexNarration(
                    "刚跑完一轮测试，核心链路有结果了。",
                    importance="high",
                    evidence=[_test_evidence(output) or command],
                    dedupe_key=f"test_completed:{_command_key(command)}",
                )
            return CodexNarration(
                "刚完成一个命令步骤，我会继续顺着结果往下查。",
                evidence=[command] if command else [],
                dedupe_key=f"command_completed:{_command_key(command)}",
            )
        if method == "item/completed" and item_type == "fileChange":
            files = _changed_files(item)
            return CodexNarration(
                "我已经落了一批文件修改，后面会继续验证它们。",
                importance="high",
                evidence=[", ".join(files[:3])] if files else [],
                dedupe_key="file_changed:" + ",".join(files[:3]),
            )
        if method == "item/completed" and item_type in {"mcpToolCall", "dynamicToolCall"}:
            tool = str(item.get("name") or item.get("toolName") or item_type)
            return CodexNarration(
                "刚完成一个工具步骤，我会根据结果继续推进。",
                evidence=[tool] if tool else [],
                dedupe_key=f"tool_completed:{tool}",
            )
        if method == "item/completed" and item_type == "agentMessage":
            # Final model text is delivered by the normal result path; avoid
            # echoing model prose as progress noise.
            return None
        if event_type == "progress.tool_completed":
            count = int(payload.get("tool_iterations") or 0)
            return CodexNarration(
                "我刚完成一个工具步骤，正在把结果接到下一步。",
                evidence=[f"已完成 {count} 个工具步骤"] if count else [],
                dedupe_key=f"tool_completed:{count}",
            )
        return None

    def status_text(
        self,
        events: list[CodexRuntimeEvent],
        *,
        workspace: str = "",
        thread_id: str = "",
    ) -> str:
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
            return f"最近有 Codex 活动，但没有需要打扰你的关键节点。\n最后活动：{age}前"
        return ""

    def format_event_line(self, event: CodexRuntimeEvent) -> str:
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
        return "codex.notification"
    if stage == "server_request":
        return "approval.requested"
    if stage:
        return f"progress.{stage}"
    return "progress.unknown"


def _event_item(notification: dict[str, Any]) -> dict[str, Any]:
    params = notification.get("params") if isinstance(notification, dict) else {}
    params = params if isinstance(params, dict) else {}
    item = params.get("item")
    return item if isinstance(item, dict) else {}


def _approval_command(payload: dict[str, Any]) -> str:
    request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    command = params.get("command") or payload.get("command") or ""
    return _one_line(str(command), 120)


def _command_preview(item: dict[str, Any]) -> str:
    command = str(item.get("command") or "")
    cwd = str(item.get("cwd") or "")
    preview = _one_line(command, 140)
    if cwd:
        return f"{preview} @ {_workspace_name(cwd)}" if preview else f"工作区 {_workspace_name(cwd)}"
    return preview


def _command_key(command: str) -> str:
    return re.sub(r"\s+", " ", command.strip())[:80]


def _changed_files(item: dict[str, Any]) -> list[str]:
    files: list[str] = []
    for change in item.get("changes") or []:
        if not isinstance(change, dict):
            continue
        path = str(change.get("path") or "")
        if path:
            files.append(path)
    return files


def _looks_like_test_command(command: str) -> bool:
    lowered = command.lower()
    return any(marker in lowered for marker in ("pytest", "npm test", "pnpm test", "cargo test", "go test", "vitest"))


def _looks_like_test_output(output: str) -> bool:
    lowered = output.lower()
    return " passed" in lowered or " failed" in lowered or " tests passed" in lowered


def _test_evidence(output: str) -> str:
    lines = [_one_line(line, 120) for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        if " passed" in lowered or " failed" in lowered or "error" in lowered:
            return line
    return lines[-1] if lines else ""


def _workspace_name(workspace: str) -> str:
    value = str(workspace or "").rstrip(os.sep)
    return os.path.basename(value) if value else "(unknown)"


def _one_line(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) > limit:
        return value[: limit - 3] + "..."
    return value


def _localize_runtime_error(text: str) -> str:
    value = re.sub(
        r"codex went silent for ([0-9.]+)s after a tool result; retiring app-server session\.",
        r"Codex app-server 在工具步骤后 \1 秒没有新事件；已回收本轮运行时。",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    value = value.replace(
        "codex app-server subprocess exited unexpectedly",
        "Codex app-server 子进程意外退出",
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
