"""Shared Claude runtime contracts used by CLI and SDK transports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol


@dataclass
class TurnResult:
    """Result of one user->assistant turn through a Claude runtime."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    turn_id: Optional[str] = None
    thread_id: Optional[str] = None
    session_id: Optional[str] = None
    token_usage_last: Optional[dict[str, Any]] = None
    token_usage_total: Optional[dict[str, Any]] = None
    should_retire: bool = False
    warning: Optional[str] = None
    error_kind: str = ""
    exit_status: Optional[int] = None
    api_retry_count: int = 0
    raw_output_tail: list[str] = field(default_factory=list)
    runtime: str = ""
    runtime_profile: str = ""
    started: bool = False
    fallback_runtime: str = ""
    fallback_reason: str = ""


class ClaudeRuntimeSession(Protocol):
    cwd: str
    thread_id: str
    session_id: str

    def ensure_started(self) -> str:
        ...

    def run_turn(self, user_input: Any, **options: Any) -> TurnResult:
        ...

    def request_interrupt(self) -> None:
        ...

    def close(self) -> None:
        ...

    def set_approval_callback(
        self,
        callback: Optional[Callable[..., str]],
    ) -> None:
        ...
