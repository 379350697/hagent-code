"""Task record construction for platform-neutral Claude control."""

from __future__ import annotations


def make_task_record(
    *,
    task_id: str,
    task_key: str,
    status: str,
    workspace: str,
    thread_id: str,
    turn_id: str,
    model: str,
    permission_mode: str,
    prompt: str,
    last_message: str,
    title: str = "",
):
    from .registry import ClaudeTaskRecord

    return ClaudeTaskRecord(
        task_id=task_id,
        task_key=task_key,
        status=status,
        workspace=workspace,
        thread_id=thread_id,
        turn_id=turn_id,
        model=model,
        permission_mode=permission_mode,
        title=(title or " ".join(prompt.strip().split()))[:80],
        last_message=last_message,
    )
