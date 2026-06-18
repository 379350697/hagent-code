"""Platform-neutral /codex command service."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from agent.transports.codex_app_server_session import CodexAppServerSession

from .formatting import (
    format_failure,
    format_run_failure,
    format_run_success,
    format_task_status,
)
from .execution import run_blocking
from .models import CommandRequest, CommandResult
from .records import make_task_record
from .runtime_config import (
    PLAN_PROMPT_PREFIX,
    load_codex_cfg,
    read_codex_config_model,
)
from .session_pool import CodexSessionPool, LiveSession
from .task_keys import build_codex_task_key, task_id_for

logger = logging.getLogger("gateway.run")


class CodexCommandService:
    """Platform-neutral implementation of the gateway /codex command."""

    def __init__(
        self,
        *,
        registry: Any = None,
        session_factory: Optional[Callable[..., CodexAppServerSession]] = None,
    ) -> None:
        if registry is None:
            from tools.codex_app_server import CodexTaskRegistry

            registry = CodexTaskRegistry()
        self._registry = registry
        self._sessions = CodexSessionPool(session_factory=session_factory)

    async def handle(self, request: CommandRequest) -> CommandResult:
        raw_args = (request.text or "").strip()
        parts = raw_args.split(maxsplit=1)
        subcommand = parts[0].lower() if parts else "status"
        rest = parts[1] if len(parts) > 1 else ""
        task_key = build_codex_task_key(request)

        if subcommand in {"status"}:
            return self._status(task_key)
        if subcommand in {"sessions", "tasks", "ls", "list"}:
            return self._sessions_status(request)
        if subcommand in {"diff", "git"}:
            return self._diff(task_key, request.workspace)
        if subcommand in {"stop", "interrupt", "cancel"}:
            return self._stop(task_key)
        if subcommand in {"permissions", "permission", "perms"}:
            return self._permissions(rest)
        if subcommand in {"steer", "补充"}:
            return CommandResult(
                "Codex steer is only available while the live Codex turn is running; "
                "send a normal follow-up with `/codex continue <instruction>`.",
                status="unsupported",
            )
        if subcommand in {"plan", "run", "new"}:
            prompt = rest.strip()
            if not prompt:
                return CommandResult(f"Usage: /codex {subcommand} <task>", status="usage")
            if subcommand == "plan":
                prompt = PLAN_PROMPT_PREFIX + prompt
            return await self._run(
                task_key,
                prompt,
                workspace=request.workspace,
                new_session=True,
                plan_mode=(subcommand == "plan"),
            )
        if subcommand in {"continue", "接着"}:
            prompt = rest.strip()
            if not prompt:
                return CommandResult("Usage: /codex continue <task>", status="usage")
            return await self._run(
                task_key,
                prompt,
                workspace=request.workspace,
                new_session=False,
                plan_mode=False,
            )

        return CommandResult(
            "Usage: /codex new <task> | continue <task> | status | sessions | "
            "diff | stop | plan <task> | permissions <auto|workspace|readonly|danger>",
            status="usage",
        )

    def _status(self, task_key: str) -> CommandResult:
        record = self._registry.get(task_key=task_key)
        if record is None:
            return CommandResult("Codex: no task found.", status="not_found")
        return CommandResult(
            format_task_status(record),
            status=str(getattr(record, "status", "unknown") or "unknown"),
            task_id=str(getattr(record, "task_id", "") or ""),
            thread_id=str(getattr(record, "thread_id", "") or ""),
        )

    def _sessions_status(self, request: CommandRequest) -> CommandResult:
        platform_prefix = f"{(request.platform or 'unknown').lower()}:"
        records = [
            record
            for record in self._registry.list(limit=50)
            if str(getattr(record, "task_key", "")).startswith(platform_prefix)
        ][:10]
        if not records:
            return CommandResult("Codex: no sessions found for this platform.", status="not_found")
        lines = ["Codex sessions"]
        for record in records:
            title = getattr(record, "title", "") or "(untitled)"
            lines.append(
                f"- {getattr(record, 'task_id', '')}: {getattr(record, 'status', '')} "
                f"{getattr(record, 'thread_id', '')} {title}"
            )
        return CommandResult("\n".join(lines), status="ok")

    def _diff(self, task_key: str, workspace: str) -> CommandResult:
        record = self._registry.get(task_key=task_key)
        resolved_workspace = workspace or (getattr(record, "workspace", "") if record else "")
        try:
            from tools.codex_app_server import _format_git_digest, _git_digest

            digest = _git_digest(resolved_workspace or os.getcwd())
            return CommandResult(
                _format_git_digest(digest),
                status="ok" if digest.get("available") else "failed",
                diagnostics={"git": digest},
            )
        except Exception as exc:
            return CommandResult(format_failure("Codex app-server failed", exc), status="failed")

    def _stop(self, task_key: str) -> CommandResult:
        live = self._sessions.peek(task_key)
        if live is None:
            return CommandResult(
                "Codex stop failed: no live Codex task for this chat.",
                status="not_found",
            )
        try:
            live.session.request_interrupt()
            if live.active_task_id:
                self._registry.update(
                    live.active_task_id,
                    status="interrupted",
                    completed_at=time.time(),
                    last_message="Codex stop requested.",
                )
            return CommandResult("Codex stop requested.", status="interrupted")
        except Exception as exc:
            return CommandResult(format_failure("Codex app-server failed", exc), status="failed")

    def _permissions(self, raw_value: str) -> CommandResult:
        value = raw_value.strip().lower()
        if not value:
            return CommandResult(
                "Codex permissions: auto, workspace, readonly, danger.\n"
                "Use `/codex permissions workspace` for the safe default.",
                status="usage",
            )
        mapping = {
            "auto": ("workspace-write", "on-request"),
            "workspace": ("workspace-write", "on-request"),
            "safe": ("workspace-write", "on-request"),
            "readonly": ("read-only", "on-request"),
            "read-only": ("read-only", "on-request"),
            "danger": ("danger-full-access", "never"),
            "yolo": ("danger-full-access", "never"),
        }
        if value not in mapping:
            return CommandResult(
                "Unknown Codex permission mode. Use auto, workspace, readonly, or danger.",
                status="usage",
            )
        sandbox, approval = mapping[value]
        try:
            from cli import save_config_value

            save_config_value("codex_app_server.sandbox", sandbox)
            save_config_value("codex_app_server.approval_policy", approval)
        except Exception:
            logger.debug("Could not persist codex permission mode", exc_info=True)
        return CommandResult(f"Codex permissions set to {sandbox} / {approval}.", status="ok")

    async def _run(
        self,
        task_key: str,
        prompt: str,
        *,
        workspace: str,
        new_session: bool,
        plan_mode: bool,
    ) -> CommandResult:
        return await run_blocking(
            self._run_sync,
            task_key,
            prompt,
            workspace or os.getcwd(),
            new_session,
            plan_mode,
        )

    def _run_sync(
        self,
        task_key: str,
        prompt: str,
        workspace: str,
        new_session: bool,
        plan_mode: bool,
    ) -> CommandResult:
        live = self._sessions.get(task_key, workspace, new_session=new_session)
        if live is None:
            return CommandResult(
                "No live Codex app-server session found for this chat. "
                "Use `/codex new <task>` to start a fresh session.",
                status="not_found",
            )
        if not live.lock.acquire(blocking=False):
            return CommandResult(
                "Codex app-server failed: a Codex task is already running for this chat. "
                "Use `/codex stop` or wait for it to finish.",
                status="busy",
            )
        try:
            return self._run_locked(live, task_key, prompt, workspace, plan_mode)
        finally:
            live.active_task_id = ""
            live.lock.release()

    def _run_locked(
        self,
        live: LiveSession,
        task_key: str,
        prompt: str,
        workspace: str,
        plan_mode: bool,
    ) -> CommandResult:
        codex_cfg = load_codex_cfg()
        model = str(codex_cfg.get("model") or read_codex_config_model())
        sandbox = str(codex_cfg.get("sandbox") or "workspace-write")
        approval = str(codex_cfg.get("approval_policy") or "on-request")
        try:
            thread_id = live.session.ensure_started()
        except Exception as exc:
            self._sessions.retire(task_key, live)
            return CommandResult(
                format_failure("Codex app-server failed", exc),
                status="failed",
                diagnostics={"phase": "thread/start"},
            )

        task_id = task_id_for(thread_id)
        live.active_task_id = task_id
        self._registry.upsert(
            make_task_record(
                task_id=task_id,
                task_key=task_key,
                status="planning" if plan_mode else "running",
                workspace=workspace,
                thread_id=thread_id,
                turn_id="",
                model=model,
                approval=approval,
                sandbox=sandbox,
                plan_mode=plan_mode,
                prompt=prompt,
                last_message="Codex: turn started",
            )
        )

        turn = live.session.run_turn(user_input=prompt)
        status = "completed"
        last_message = "Codex: turn completed"
        if getattr(turn, "error", None):
            status = "failed"
            last_message = str(turn.error)
        elif getattr(turn, "interrupted", False):
            status = "interrupted"
            last_message = "Codex: turn interrupted"

        self._registry.update(
            task_id,
            status=status,
            turn_id=getattr(turn, "turn_id", None) or "",
            completed_at=time.time(),
            last_message=last_message,
            token_usage=getattr(turn, "token_usage_total", None) or {},
        )
        if getattr(turn, "should_retire", False):
            self._sessions.retire(task_key, live)

        text = getattr(turn, "final_text", "") or ""
        output = (
            format_run_failure(workspace, thread_id, str(turn.error))
            if getattr(turn, "error", None)
            else format_run_success(workspace, thread_id, text)
        )
        return CommandResult(
            output,
            status=status,
            task_id=task_id,
            thread_id=thread_id,
            diagnostics={"turn_id": getattr(turn, "turn_id", None) or ""},
        )

_DEFAULT_SERVICE: CodexCommandService | None = None
_DEFAULT_SERVICE_LOCK = threading.Lock()


def get_codex_command_service() -> CodexCommandService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        with _DEFAULT_SERVICE_LOCK:
            if _DEFAULT_SERVICE is None:
                _DEFAULT_SERVICE = CodexCommandService()
    return _DEFAULT_SERVICE
