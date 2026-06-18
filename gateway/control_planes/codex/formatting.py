"""User-facing formatting for Codex command results."""

from __future__ import annotations

from typing import Any


def format_failure(prefix: str, error: Any) -> str:
    message = str(error or "unknown error")
    lowered = message.lower()
    if "codex approval bridge unavailable" in lowered:
        return message
    if "codex approval denied" in lowered or "codex approval timed out" in lowered:
        return message
    if "bwrap" in lowered or "bubblewrap" in lowered or "sandbox" in lowered:
        return f"Codex sandbox failed: {message}"
    if "timed out" in lowered or "timeout" in lowered:
        return f"Codex task timed out: {message}"
    return f"{prefix}: {message}"


def format_task_status(record: Any) -> str:
    lines = [
        f"Codex task {getattr(record, 'task_id', '')}: {getattr(record, 'status', '')}",
        f"Thread: {getattr(record, 'thread_id', '') or 'unknown'}",
    ]
    turn_id = getattr(record, "turn_id", "")
    if turn_id:
        lines.append(f"Turn: {turn_id}")
    workspace = getattr(record, "workspace", "")
    if workspace:
        lines.append(f"Workspace: {workspace}")
    lines.append(
        "Permissions: "
        f"{getattr(record, 'sandbox', 'workspace-write')} / "
        f"{getattr(record, 'approval_policy', 'on-request')}"
    )
    model = getattr(record, "model", "")
    if model:
        lines.append(f"Model: {model}")
    title = getattr(record, "title", "")
    if title:
        lines.append(f"Task: {title}")
    last = getattr(record, "last_message", "")
    if last:
        lines.append(f"Last: {last}")
    return "\n".join(lines)


def format_run_success(workspace: str, thread_id: str, message: str) -> str:
    body = message.strip() or "(Codex completed without a final message.)"
    return (
        "Codex result\n"
        f"Workspace: {workspace} · Session: {thread_id}\n\n"
        f"Codex said:\n{body}"
    )


def format_run_failure(workspace: str, thread_id: str, error: str) -> str:
    return (
        "Codex task failed\n"
        f"Workspace: {workspace} · Session: {thread_id or 'new'}\n"
        f"Error: {format_failure('Codex app-server failed', error)}"
    )
