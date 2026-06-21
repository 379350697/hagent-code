"""Repair helpers for Codex control-plane records."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RecoverableTurn:
    task_id: str
    task_key: str
    thread_id: str
    turn_id: str
    workspace: str
    message_preview: str


def find_recoverable_completed_turns(
    registry: Any,
    *,
    codex_home: str = "",
    task_key: str = "",
    limit: int = 50,
) -> list[RecoverableTurn]:
    try:
        records = registry.list(task_key=task_key or None, limit=limit)
    except TypeError:
        records = registry.list()
    except Exception:
        records = []
    recoverable: list[RecoverableTurn] = []
    for record in records:
        status = str(getattr(record, "status", "") or "")
        if status not in {"failed", "unconfirmed"}:
            continue
        thread_id = str(getattr(record, "thread_id", "") or "")
        turn_id = str(getattr(record, "turn_id", "") or "")
        if not thread_id or not turn_id:
            continue
        complete = latest_native_task_complete(
            thread_id,
            codex_home=codex_home,
            expected_turn_id=turn_id,
        )
        if complete is None:
            continue
        native_turn_id, message = complete
        recoverable.append(
            RecoverableTurn(
                task_id=str(getattr(record, "task_id", "") or ""),
                task_key=str(getattr(record, "task_key", "") or ""),
                thread_id=thread_id,
                turn_id=native_turn_id,
                workspace=str(getattr(record, "workspace", "") or ""),
                message_preview=" ".join(message.split())[:160],
            )
        )
    return recoverable


def apply_recovered_turns(registry: Any, turns: list[RecoverableTurn]) -> int:
    count = 0
    for turn in turns:
        if not turn.task_id:
            continue
        updated = registry.update(
            turn.task_id,
            status="completed",
            turn_id=turn.turn_id,
            last_message="Codex: turn completed",
        )
        if updated is not None:
            count += 1
    return count


def latest_native_task_complete(
    thread_id: str,
    *,
    codex_home: str = "",
    expected_turn_id: str = "",
    since_epoch: float = 0.0,
) -> tuple[str, str] | None:
    if not thread_id:
        return None
    home = codex_home or os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    pattern = os.path.join(home, "sessions", "**", f"*{thread_id}.jsonl")
    best_time = 0.0
    best: tuple[str, str] | None = None
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("type") != "event_msg":
                        continue
                    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
                    if payload.get("type") != "task_complete":
                        continue
                    turn_id = str(payload.get("turn_id") or "")
                    if expected_turn_id and turn_id != expected_turn_id:
                        continue
                    message = str(payload.get("last_agent_message") or "").strip()
                    if not message:
                        continue
                    timestamp = _parse_timestamp(record.get("timestamp"))
                    if since_epoch and (
                        not timestamp or timestamp < since_epoch - 5.0
                    ):
                        continue
                    if timestamp >= best_time:
                        best_time = timestamp
                        best = (turn_id, message)
        except OSError:
            continue
    return best


def _parse_timestamp(value: Any) -> float:
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
