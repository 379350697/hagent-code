"""Task record construction for platform-neutral Codex control."""

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
    approval: str,
    sandbox: str,
    plan_mode: bool,
    prompt: str,
    last_message: str,
    title: str = "",
):
    from .registry import CodexTaskRecord

    return CodexTaskRecord(
        task_id=task_id,
        task_key=task_key,
        status=status,
        workspace=workspace,
        thread_id=thread_id,
        turn_id=turn_id,
        model=model,
        approval_policy=approval,
        sandbox=sandbox,
        plan_mode=plan_mode,
        plan_first="off",
        title=(title or " ".join(_strip_plan_prefix(prompt).strip().split()))[:80],
        last_message=last_message,
    )


def _strip_plan_prefix(prompt: str) -> str:
    marker = "confirms.\n\n"
    if prompt.startswith("Create a detailed implementation plan first."):
        return prompt.split(marker, 1)[1] if marker in prompt else prompt
    return prompt
