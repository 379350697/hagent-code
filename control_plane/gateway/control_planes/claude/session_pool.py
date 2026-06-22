"""Live Claude runtime session pool scoped by platform/chat/thread keys."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent.transports.claude_runtime import ClaudeRuntimeSession
from agent.transports.claude_runtime_factory import create_claude_runtime_session

logger = logging.getLogger("gateway.run")


@dataclass
class LiveSession:
    session: ClaudeRuntimeSession
    workspace: str
    config_overrides: list[str]
    runtime: str
    runtime_fallback: str
    sdk_profile: str
    lock: threading.Lock
    thread_id: str = ""
    active_task_id: str = ""


class ClaudeSessionPool:
    """Owns live Claude CLI sessions without knowing platform adapters."""

    def __init__(
        self,
        *,
        session_factory: Optional[Callable[..., ClaudeRuntimeSession]] = None,
    ) -> None:
        self._session_factory = session_factory or create_claude_runtime_session
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.RLock()

    def get(
        self,
        task_key: str,
        workspace: str,
        *,
        new_session: bool,
        config_overrides: list[str] | None = None,
        resume_thread_id: str = "",
        permission_mode: str = "acceptEdits",
        model: str = "",
        effort: str = "",
        runtime: str = "agent_sdk",
        runtime_fallback: str = "cli",
        sdk_profile: str = "",
        sdk_profile_config: dict[str, Any] | None = None,
    ) -> LiveSession | None:
        desired_config = list(config_overrides or [])
        desired_thread_id = str(resume_thread_id or "")
        desired_runtime = str(runtime or "agent_sdk")
        desired_fallback = str(runtime_fallback or "")
        desired_profile = str(sdk_profile or "")
        with self._lock:
            existing = self._sessions.get(task_key)
            retired_existing = False
            if (
                existing is not None
                and (
                    new_session
                    or existing.workspace != workspace
                    or existing.config_overrides != desired_config
                    or existing.runtime != desired_runtime
                    or existing.runtime_fallback != desired_fallback
                    or existing.sdk_profile != desired_profile
                    or (
                        desired_thread_id
                        and existing.thread_id
                        and existing.thread_id != desired_thread_id
                    )
                    or (desired_thread_id and not existing.thread_id)
                )
            ):
                self.retire(task_key, existing)
                existing = None
                retired_existing = True
            if existing is not None:
                return existing
            if not new_session and not retired_existing and not desired_thread_id:
                return None
            session = self._session_factory(
                cwd=workspace,
                config_overrides=desired_config,
                resume_thread_id=desired_thread_id,
                permission_mode=permission_mode,
                model=model,
                effort=effort,
                runtime=desired_runtime,
                runtime_fallback=desired_fallback,
                sdk_profile_config=sdk_profile_config or {},
            )
            live = LiveSession(
                session=session,
                workspace=workspace,
                config_overrides=desired_config,
                runtime=desired_runtime,
                runtime_fallback=desired_fallback,
                sdk_profile=desired_profile,
                lock=threading.Lock(),
                thread_id=desired_thread_id,
            )
            self._sessions[task_key] = live
            return live

    def peek(self, task_key: str) -> LiveSession | None:
        with self._lock:
            return self._sessions.get(task_key)

    def retire(self, task_key: str, live: LiveSession) -> None:
        with self._lock:
            if self._sessions.get(task_key) is live:
                self._sessions.pop(task_key, None)
        try:
            live.session.close()
        except Exception:
            logger.debug("claude session retire close failed", exc_info=True)
