"""Live Codex app-server session pool scoped by platform/chat/thread keys."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from agent.transports.codex_app_server_session import CodexAppServerSession

from .runtime_config import get_approval_callback

logger = logging.getLogger("gateway.run")


@dataclass
class LiveSession:
    session: CodexAppServerSession
    workspace: str
    lock: threading.Lock
    active_task_id: str = ""


class CodexSessionPool:
    """Owns live Codex app-server sessions without knowing platform adapters."""

    def __init__(
        self,
        *,
        session_factory: Optional[Callable[..., CodexAppServerSession]] = None,
    ) -> None:
        self._session_factory = session_factory or CodexAppServerSession
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.RLock()

    def get(
        self,
        task_key: str,
        workspace: str,
        *,
        new_session: bool,
    ) -> LiveSession | None:
        with self._lock:
            existing = self._sessions.get(task_key)
            if new_session and existing is not None:
                self.retire(task_key, existing)
                existing = None
            if existing is not None:
                return existing
            if not new_session:
                return None
            session = self._session_factory(
                cwd=workspace,
                approval_callback=get_approval_callback(),
            )
            live = LiveSession(session=session, workspace=workspace, lock=threading.Lock())
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
            logger.debug("codex session retire close failed", exc_info=True)
