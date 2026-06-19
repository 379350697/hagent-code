"""Persistent selected Codex session mapping."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import threading
import time
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is unavailable on Windows.
    fcntl = None


SELECTION_VERSION = 1


def codex_selection_path() -> str:
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(home, "codex-control-plane", "selected_sessions.json")


@dataclass
class SelectedSession:
    task_id: str
    thread_id: str
    workspace: str = ""
    selected_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "threadId": self.thread_id,
            "workspace": self.workspace,
            "selectedAt": self.selected_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelectedSession":
        return cls(
            task_id=str(data.get("taskId") or data.get("task_id") or ""),
            thread_id=str(data.get("threadId") or data.get("thread_id") or ""),
            workspace=str(data.get("workspace") or ""),
            selected_at=float(data.get("selectedAt") or data.get("selected_at") or 0.0),
        )


class SelectedSessionStore:
    """Stores the current Codex session selected for each platform/chat/thread."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or codex_selection_path()
        self._lock = threading.RLock()

    def get(self, task_key: str) -> SelectedSession | None:
        with self._lock:
            with self._file_lock_locked(exclusive=False):
                selections = self._load_locked()
        raw = selections.get(task_key)
        if not isinstance(raw, dict):
            return None
        selected = SelectedSession.from_dict(raw)
        if not selected.task_id and not selected.thread_id:
            return None
        return selected

    def set(
        self,
        task_key: str,
        *,
        task_id: str,
        thread_id: str,
        workspace: str = "",
    ) -> SelectedSession:
        selected = SelectedSession(
            task_id=task_id,
            thread_id=thread_id,
            workspace=workspace,
            selected_at=time.time(),
        )
        with self._lock:
            with self._file_lock_locked():
                selections = self._load_locked()
                selections[task_key] = selected.to_dict()
                self._save_locked(selections)
        return selected

    def clear(self, task_key: str) -> None:
        with self._lock:
            with self._file_lock_locked():
                selections = self._load_locked()
                selections.pop(task_key, None)
                self._save_locked(selections)

    @contextmanager
    def _file_lock_locked(self, *, exclusive: bool = True):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        lock_path = f"{self.path}.lock"
        with open(lock_path, "a+", encoding="utf-8") as lock_handle:
            if fcntl is not None:
                operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                fcntl.flock(lock_handle.fileno(), operation)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _load_locked(self) -> dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return {}
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        selections = payload.get("selections")
        return selections if isinstance(selections, dict) else {}

    def _save_locked(self, selections: dict[str, Any]) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "version": SELECTION_VERSION,
            "selections": selections,
        }
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
