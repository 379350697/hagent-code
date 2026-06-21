"""Persistent task registry for the Codex control plane."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
import threading
import time
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is unavailable on Windows.
    fcntl = None


JsonDict = dict[str, Any]
SAFE_APPROVAL_POLICY = "on-request"
SAFE_SANDBOX = "workspace-write"
TASK_REGISTRY_VERSION = 1
MAX_RECENT_EVENTS = 30


def codex_registry_path() -> str:
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(home, "codex-control-plane", "tasks.json")


def _event_digest(event: JsonDict) -> JsonDict:
    digest: JsonDict = {"at": time.time(), "type": event.get("type")}
    method = event.get("method")
    if method:
        digest["method"] = method
    message = event.get("displayMessage") or event.get("message")
    if message:
        digest["message"] = message
    return {k: v for k, v in digest.items() if v not in (None, "")}


@dataclass
class CodexTaskRecord:
    task_id: str
    task_key: str = "default"
    status: str = "starting"
    workspace: str = ""
    thread_id: str = ""
    turn_id: str = ""
    model: str = ""
    reasoning_effort: str = ""
    approval_policy: str = SAFE_APPROVAL_POLICY
    sandbox: str = SAFE_SANDBOX
    plan_mode: bool = False
    plan_first: str = "off"
    title: str = ""
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turn_started_at: float | None = None
    completed_at: float | None = None
    last_message: str = ""
    token_usage: JsonDict = field(default_factory=dict)
    recent_events: list[JsonDict] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {
            "task_id": self.task_id,
            "task_key": self.task_key,
            "status": self.status,
            "workspace": self.workspace,
            "threadId": self.thread_id,
            "turnId": self.turn_id,
            "model": self.model,
            "reasoningEffort": self.reasoning_effort,
            "approvalPolicy": self.approval_policy,
            "sandbox": self.sandbox,
            "planMode": self.plan_mode,
            "planFirst": self.plan_first,
            "title": self.title,
            "startedAt": self.started_at,
            "updatedAt": self.updated_at,
            "turnStartedAt": self.turn_started_at,
            "completedAt": self.completed_at,
            "lastMessage": self.last_message,
            "tokenUsage": self.token_usage,
            "recentEvents": self.recent_events[-MAX_RECENT_EVENTS:],
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "CodexTaskRecord":
        return cls(
            task_id=str(data.get("task_id") or data.get("taskId") or ""),
            task_key=str(data.get("task_key") or data.get("taskKey") or "default"),
            status=str(data.get("status") or "unknown"),
            workspace=str(data.get("workspace") or ""),
            thread_id=str(data.get("threadId") or data.get("thread_id") or ""),
            turn_id=str(data.get("turnId") or data.get("turn_id") or ""),
            model=str(data.get("model") or ""),
            reasoning_effort=str(data.get("reasoningEffort") or data.get("reasoning_effort") or ""),
            approval_policy=str(data.get("approvalPolicy") or SAFE_APPROVAL_POLICY),
            sandbox=str(data.get("sandbox") or SAFE_SANDBOX),
            plan_mode=bool(data.get("planMode")),
            plan_first=str(data.get("planFirst") or "off"),
            title=str(data.get("title") or ""),
            started_at=float(data.get("startedAt") or time.time()),
            updated_at=float(data.get("updatedAt") or time.time()),
            turn_started_at=(
                float(data["turnStartedAt"]) if data.get("turnStartedAt") else None
            ),
            completed_at=(float(data["completedAt"]) if data.get("completedAt") else None),
            last_message=str(data.get("lastMessage") or ""),
            token_usage=data.get("tokenUsage") if isinstance(data.get("tokenUsage"), dict) else {},
            recent_events=(
                data.get("recentEvents") if isinstance(data.get("recentEvents"), list) else []
            ),
        )


class CodexTaskRegistry:
    """Small persistent index of Codex work controlled by Hermes."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or codex_registry_path()
        self._lock = threading.RLock()
        self._loaded = False
        self._records: dict[str, CodexTaskRecord] = {}
        self._latest_by_key: dict[str, str] = {}

    def upsert(self, record: CodexTaskRecord) -> CodexTaskRecord:
        with self._lock:
            with self._file_lock_locked():
                self._load_locked(force=True)
                record.updated_at = time.time()
                self._records[record.task_id] = record
                self._latest_by_key[record.task_key] = record.task_id
                self._save_locked()
            return record

    def update(self, task_id: str, **fields: Any) -> CodexTaskRecord | None:
        with self._lock:
            with self._file_lock_locked():
                self._load_locked(force=True)
                record = self._records.get(task_id)
                if record is None:
                    return None
                for key, value in fields.items():
                    if hasattr(record, key):
                        setattr(record, key, value)
                record.updated_at = time.time()
                self._latest_by_key[record.task_key] = record.task_id
                self._save_locked()
                return record

    def record_event(self, task_id: str, event: JsonDict) -> None:
        with self._lock:
            with self._file_lock_locked():
                self._load_locked(force=True)
                record = self._records.get(task_id)
                if record is None:
                    return
                record.recent_events.append(_event_digest(event))
                record.recent_events = record.recent_events[-MAX_RECENT_EVENTS:]
                message = event.get("displayMessage") or event.get("message")
                if message:
                    record.last_message = str(message)
                record.updated_at = time.time()
                self._save_locked()

    def get(
        self,
        task_id: str | None = None,
        *,
        task_key: str | None = None,
        thread_id: str | None = None,
    ) -> CodexTaskRecord | None:
        with self._lock:
            with self._file_lock_locked(exclusive=False):
                self._load_locked(force=True)
            if task_id and task_id in self._records:
                return self._records[task_id]
            if thread_id:
                for record in self._records.values():
                    if record.thread_id == thread_id:
                        return record
            key = task_key or "default"
            latest_id = self._latest_by_key.get(key)
            if latest_id:
                return self._records.get(latest_id)
            return None

    def list(self, *, task_key: str | None = None, limit: int = 10) -> list[CodexTaskRecord]:
        with self._lock:
            with self._file_lock_locked(exclusive=False):
                self._load_locked(force=True)
            records = list(self._records.values())
            if task_key:
                records = [record for record in records if record.task_key == task_key]
            records.sort(key=lambda item: item.updated_at, reverse=True)
            return records[: max(1, limit)]

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

    def _load_locked(self, *, force: bool = False) -> None:
        if self._loaded and not force:
            return
        self._loaded = True
        if force:
            self._records = {}
            self._latest_by_key = {}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        records = payload.get("tasks")
        if not isinstance(records, list):
            return
        loaded: list[CodexTaskRecord] = []
        for raw in records:
            if not isinstance(raw, dict):
                continue
            record = CodexTaskRecord.from_dict(raw)
            if record.task_id:
                loaded.append(record)
        loaded.sort(key=lambda item: item.updated_at, reverse=True)
        for record in loaded:
            self._records[record.task_id] = record
            self._latest_by_key.setdefault(record.task_key, record.task_id)

    def _save_locked(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        records = sorted(
            self._records.values(),
            key=lambda item: item.updated_at,
            reverse=True,
        )
        payload = {
            "version": TASK_REGISTRY_VERSION,
            "tasks": [record.to_dict() for record in records],
        }
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
