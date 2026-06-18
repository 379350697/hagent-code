"""Data contracts for platform-neutral Codex command handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommandRequest:
    platform: str
    chat_id: str
    user_id: str = ""
    thread_id: str = ""
    text: str = ""
    workspace: str = ""


@dataclass(frozen=True)
class CommandResult:
    text: str
    status: str = "ok"
    task_id: str = ""
    thread_id: str = ""
    diagnostics: dict[str, Any] | None = None
