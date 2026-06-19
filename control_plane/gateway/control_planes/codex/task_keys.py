"""Task identity helpers for platform-isolated Codex sessions."""

from __future__ import annotations

import hashlib
import time

from .models import CommandRequest


def build_codex_task_key(request: CommandRequest) -> str:
    """Return the platform/chat/thread scoped key used for Codex tasks."""

    platform = (request.platform or "unknown").strip().lower() or "unknown"
    chat_id = str(request.chat_id or request.user_id or "unknown").strip() or "unknown"
    thread_id = str(request.thread_id or "main").strip() or "main"
    return f"{platform}:{chat_id}:{thread_id}"


def task_id_for(thread_id: str, turn_id: str = "") -> str:
    seed_value = str(turn_id) if turn_id else f"{time.time():.6f}"
    seed = f"{thread_id}:{seed_value}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
