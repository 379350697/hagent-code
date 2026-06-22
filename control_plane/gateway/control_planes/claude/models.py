"""Data contracts for platform-neutral Claude command handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class CommandRequest:
    platform: str
    chat_id: str
    user_id: str = ""
    thread_id: str = ""
    text: str = ""
    workspace: str = ""
    approval_session_key: str = ""
    approval_chat_id: str = ""
    approval_thread_metadata: dict[str, Any] = field(default_factory=dict)
    approval_notify: Callable[[dict[str, Any]], None] | None = None
    progress_notify: Callable[[dict[str, Any]], None] | None = None
    is_admin: bool = False


@dataclass(frozen=True)
class CommandResult:
    text: str
    status: str = "ok"
    task_id: str = ""
    thread_id: str = ""
    diagnostics: dict[str, Any] | None = None
