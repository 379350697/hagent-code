"""Append-only runtime events for the Claude control plane."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import re
import sqlite3
import threading
import time
from typing import Any


EVENT_STORE_VERSION = 1
MAX_PAYLOAD_STRING_LENGTH = 10_000
MAX_PAYLOAD_JSON_LENGTH = 64_000
REDACTED = "[REDACTED]"

_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b("
    r"(?:[A-Za-z0-9_]*_)?"
    r"(?:password|passwd|token|secret|api[_-]?key|access[_-]?key|private[_-]?key|credential|authorization|auth)"
    r"(?:_[A-Za-z0-9_]*)?"
    r")\s*([=:])\s*([\"']?)([^\s\"'`;,]+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\b(Bearer)\s+([A-Za-z0-9._~+/\-]+=*)", re.IGNORECASE)
_AUTH_HEADER_RE = re.compile(
    r"\b(Authorization)\s*:\s*(?:Bearer\s+)?([^\s;,]+)",
    re.IGNORECASE,
)
_SK_SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{6,}\b")
_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "token",
    "secret",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "credential",
    "authorization",
)


@dataclass(frozen=True)
class ClaudeRuntimeEvent:
    id: int
    task_key: str
    task_id: str
    thread_id: str
    turn_id: str
    platform: str
    chat_id: str
    event_type: str
    payload: dict[str, Any]
    occurred_at: float


def claude_events_path() -> str:
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(home, "claude-control-plane", "events.sqlite3")


class ClaudeRuntimeEventStore:
    """Small SQLite ledger for Claude CLI runtime events."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or claude_events_path()
        self._lock = threading.RLock()
        self._migrated = False

    def append(
        self,
        *,
        task_key: str,
        task_id: str,
        thread_id: str,
        turn_id: str = "",
        platform: str = "",
        chat_id: str = "",
        event_type: str,
        payload: dict[str, Any] | None = None,
        occurred_at: float | None = None,
    ) -> ClaudeRuntimeEvent:
        safe_payload = _safe_payload(payload or {})
        payload_json = json.dumps(safe_payload, ensure_ascii=False, sort_keys=True)
        if len(payload_json) > MAX_PAYLOAD_JSON_LENGTH:
            safe_payload = {
                "truncated": True,
                "preview": payload_json[:MAX_PAYLOAD_JSON_LENGTH],
            }
            payload_json = json.dumps(safe_payload, ensure_ascii=False, sort_keys=True)
        timestamp = float(occurred_at or time.time())
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO claude_runtime_events (
                        task_key, task_id, thread_id, turn_id, platform, chat_id,
                        event_type, payload_json, occurred_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_key,
                        task_id,
                        thread_id,
                        turn_id,
                        platform,
                        chat_id,
                        event_type,
                        payload_json,
                        timestamp,
                    ),
                )
                event_id = int(cur.lastrowid)
        return ClaudeRuntimeEvent(
            id=event_id,
            task_key=task_key,
            task_id=task_id,
            thread_id=thread_id,
            turn_id=turn_id,
            platform=platform,
            chat_id=chat_id,
            event_type=event_type,
            payload=safe_payload,
            occurred_at=timestamp,
        )

    def tail(
        self,
        *,
        task_key: str = "",
        task_id: str = "",
        thread_id: str = "",
        limit: int = 50,
    ) -> list[ClaudeRuntimeEvent]:
        where: list[str] = []
        params: list[Any] = []
        if task_key:
            where.append("task_key = ?")
            params.append(task_key)
        if task_id:
            where.append("task_id = ?")
            params.append(task_id)
        if thread_id:
            where.append("thread_id = ?")
            params.append(thread_id)
        return self._query(where, params, limit=limit)

    def list_after(
        self,
        *,
        task_key: str = "",
        task_id: str = "",
        thread_id: str = "",
        after_id: int = 0,
        limit: int = 100,
    ) -> list[ClaudeRuntimeEvent]:
        where = ["id > ?"]
        params: list[Any] = [int(after_id)]
        if task_key:
            where.append("task_key = ?")
            params.append(task_key)
        if task_id:
            where.append("task_id = ?")
            params.append(task_id)
        if thread_id:
            where.append("thread_id = ?")
            params.append(thread_id)
        return self._query(where, params, limit=limit)

    def _query(
        self,
        where: list[str],
        params: list[Any],
        *,
        limit: int,
    ) -> list[ClaudeRuntimeEvent]:
        limit = max(1, int(limit))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT id, task_key, task_id, thread_id, turn_id, platform,
                           chat_id, event_type, payload_json, occurred_at
                    FROM claude_runtime_events
                    {clause}
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (*params, limit),
                ).fetchall()
        events = [_row_to_event(row) for row in rows]
        events.reverse()
        return events

    @contextmanager
    def _connect(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            if not self._migrated:
                self._migrate(conn)
                self._migrated = True
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claude_runtime_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_key TEXT NOT NULL,
                task_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                turn_id TEXT NOT NULL DEFAULT '',
                platform TEXT NOT NULL DEFAULT '',
                chat_id TEXT NOT NULL DEFAULT '',
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                occurred_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claude_events_task_key_id "
            "ON claude_runtime_events(task_key, id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claude_events_task_id_id "
            "ON claude_runtime_events(task_id, id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claude_events_thread_id_id "
            "ON claude_runtime_events(thread_id, id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claude_runtime_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO claude_runtime_meta(key, value) VALUES (?, ?)",
            ("version", str(EVENT_STORE_VERSION)),
        )


def _row_to_event(row: sqlite3.Row) -> ClaudeRuntimeEvent:
    try:
        payload = json.loads(str(row["payload_json"] or "{}"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {"value": payload}
    return ClaudeRuntimeEvent(
        id=int(row["id"]),
        task_key=str(row["task_key"] or ""),
        task_id=str(row["task_id"] or ""),
        thread_id=str(row["thread_id"] or ""),
        turn_id=str(row["turn_id"] or ""),
        platform=str(row["platform"] or ""),
        chat_id=str(row["chat_id"] or ""),
        event_type=str(row["event_type"] or ""),
        payload=payload,
        occurred_at=float(row["occurred_at"] or 0),
    )


def _safe_payload(value: Any) -> Any:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            normalized = key_text.lower().replace("-", "_")
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                safe[key_text] = REDACTED
            else:
                safe[key_text] = _safe_payload(item)
        return safe
    if isinstance(value, list):
        return [_safe_payload(item) for item in value[:100]]
    if isinstance(value, tuple):
        return [_safe_payload(item) for item in value[:100]]
    if isinstance(value, str):
        text = _redact_text(value)
        if len(text) > MAX_PAYLOAD_STRING_LENGTH:
            return text[:MAX_PAYLOAD_STRING_LENGTH] + "...<truncated>"
        return text
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _safe_payload(str(value))


def _redact_text(text: str) -> str:
    redacted = _AUTH_HEADER_RE.sub(lambda m: f"{m.group(1)}: {REDACTED}", text)
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}", redacted)
    redacted = _BEARER_RE.sub(lambda m: f"{m.group(1)} {REDACTED}", redacted)
    return _SK_SECRET_RE.sub(REDACTED, redacted)
