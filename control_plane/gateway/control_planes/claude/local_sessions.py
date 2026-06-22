"""Read-only index of Claude Code's local JSONL session store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClaudeLocalSession:
    session_id: str
    title: str
    cwd: str
    created_at: str
    updated_at: str
    source_path: str
    permission_mode: str = ""


class ClaudeLocalSessionIndex:
    """Discover sessions written by Claude Code under ``~/.claude/projects``."""

    def __init__(self, claude_home: Path | None = None) -> None:
        self._claude_home = claude_home or Path.home() / ".claude"

    def list_recent(self, *, limit: int = 50, cwd: str = "") -> list[ClaudeLocalSession]:
        wanted_cwd = str(Path(cwd).expanduser().resolve()) if cwd else ""
        sessions = []
        for path in self._session_files():
            session = self._session_from_file(path)
            if session is None:
                continue
            if wanted_cwd and session.cwd:
                try:
                    if str(Path(session.cwd).expanduser().resolve()) != wanted_cwd:
                        continue
                except OSError:
                    if session.cwd != cwd:
                        continue
            sessions.append(session)
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return sessions[: max(0, limit)]

    def get(self, session_id: str) -> ClaudeLocalSession | None:
        wanted = session_id.strip()
        if not wanted:
            return None
        for path in self._session_files():
            if path.stem == wanted:
                return self._session_from_file(path)
        for path in self._session_files():
            session = self._session_from_file(path)
            if session is not None and session.session_id == wanted:
                return session
        return None

    def _session_files(self) -> list[Path]:
        projects_dir = self._claude_home / "projects"
        if not projects_dir.exists():
            return []
        return [
            path
            for path in projects_dir.rglob("*.jsonl")
            if path.is_file() and "/subagents/" not in path.as_posix()
        ]

    def _session_from_file(self, path: Path) -> ClaudeLocalSession | None:
        session_id = path.stem
        cwd = ""
        created_at = ""
        updated_at = ""
        permission_mode = ""
        title = ""
        for row in _iter_jsonl(path):
            row_session_id = str(row.get("sessionId") or "")
            if row_session_id:
                session_id = row_session_id
            timestamp = str(row.get("timestamp") or "")
            if timestamp:
                created_at = created_at or timestamp
                updated_at = timestamp
            cwd = str(row.get("cwd") or cwd)
            permission_mode = str(row.get("permissionMode") or permission_mode)
            title = _session_title_from_row(row) or title
        if not session_id:
            return None
        fallback_time = _mtime_iso(path)
        return ClaudeLocalSession(
            session_id=session_id,
            title=title or f"Claude {session_id[:8]}",
            cwd=cwd,
            created_at=created_at or fallback_time,
            updated_at=updated_at or fallback_time,
            source_path=str(path),
            permission_mode=permission_mode,
        )


def _iter_jsonl(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
    except OSError:
        return


def _session_title_from_row(row: dict[str, Any]) -> str:
    for key in ("title", "summary"):
        title = _normalize_title(row.get(key))
        if title:
            return title
    if row.get("type") == "system" and row.get("subtype") == "away_summary":
        return _normalize_title(row.get("content"))
    if row.get("type") == "user":
        message = row.get("message")
        if isinstance(message, dict):
            return _normalize_title(_content_text(message.get("content")))
    return ""


def _normalize_title(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    title = " ".join(value.split()).strip()
    title = title.removesuffix("(disable recaps in /config)").strip()
    return _first_sentence(title)[:160].strip()


def _first_sentence(title: str) -> str:
    sentence_ends: list[int] = []
    for marker in (". ", "? ", "! "):
        index = title.find(marker)
        if index >= 0:
            sentence_ends.append(index + 1)
    for marker in ("。", "？", "！"):
        index = title.find(marker)
        if index >= 0:
            sentence_ends.append(index + 1)
    return title[: min(sentence_ends)].strip() if sentence_ends else title


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
