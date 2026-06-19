"""Platform-neutral /codex command service."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from agent.transports.codex_app_server_session import CodexAppServerSession

from .approval import CodexApprovalBridge
from .formatting import (
    format_failure,
    format_run_failure,
    format_run_success,
    format_task_status,
)
from .execution import run_blocking
from .git_digest import format_git_digest, git_digest
from .models import CommandRequest, CommandResult
from .registry import CodexTaskRegistry
from .records import make_task_record
from .runtime_config import (
    PLAN_PROMPT_PREFIX,
    codex_app_server_config_overrides,
    codex_app_server_turn_options,
    codex_permission_profiles,
    load_codex_cfg,
    normalize_permission_profile,
    normalize_sandbox_mode,
    read_codex_config_model,
)
from .selection import SelectedSessionStore
from .session_pool import CodexSessionPool, LiveSession
from .task_keys import build_codex_task_key, task_id_for
from .workspaces import (
    WorkspaceSelectionStore,
    discover_git_workspaces,
    workspace_scan_roots,
)

logger = logging.getLogger("gateway.run")


class CodexCommandService:
    """Platform-neutral implementation of the gateway /codex command."""

    def __init__(
        self,
        *,
        registry: Any = None,
        workspace_store: Any = None,
        selected_store: Any = None,
        session_factory: Optional[Callable[..., CodexAppServerSession]] = None,
    ) -> None:
        if registry is None:
            registry = CodexTaskRegistry()
        self._registry = registry
        self._workspace_store = workspace_store or WorkspaceSelectionStore()
        self._selected_store = selected_store or SelectedSessionStore()
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
            return self._sessions_status(request, task_key, rest)
        if subcommand in {"select", "use"}:
            return self._select_session(request, task_key, rest)
        if subcommand in {"resume", "接续"}:
            return await self._resume_session(request, task_key, rest)
        if subcommand in {"workspace", "workspaces", "cwd", "repo"}:
            return self._workspace_command(request, task_key, rest)
        if subcommand in {"diff", "git"}:
            return self._diff(task_key, "")
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
            new_session = subcommand in {"run", "new"}
            resume_record = None
            if not new_session and self._sessions.peek(task_key) is None:
                resume_record = self._selected_or_latest_record(task_key)
                new_session = resume_record is None
            workspace = (
                str(getattr(resume_record, "workspace", "") or "")
                if resume_record is not None
                else self._workspace_for(request, task_key)
            )
            return await self._run(
                request,
                task_key,
                prompt,
                workspace=workspace,
                new_session=new_session,
                plan_mode=(subcommand == "plan"),
                resume_thread_id=(
                    str(getattr(resume_record, "thread_id", "") or "")
                    if resume_record is not None
                    else ""
                ),
                select_session=True,
            )
        if subcommand in {"continue", "接着"}:
            prompt = rest.strip()
            if not prompt:
                return CommandResult("Usage: /codex continue <task>", status="usage")
            resume_record = self._selected_or_latest_record(task_key)
            live = self._sessions.peek(task_key)
            if resume_record is None and live is None:
                return CommandResult(
                    "No selected Codex session found for this chat. "
                    "Use `/codex new <task>` or `/codex resume <session> <task>`.",
                    status="not_found",
                )
            return await self._run(
                request,
                task_key,
                prompt,
                workspace=(
                    str(getattr(resume_record, "workspace", "") or "")
                    if resume_record is not None
                    else self._workspace_for(request, task_key)
                ),
                new_session=False,
                plan_mode=False,
                resume_thread_id=(
                    str(getattr(resume_record, "thread_id", "") or "")
                    if resume_record is not None
                    else ""
                ),
                select_session=True,
            )

        return CommandResult(
            "Usage: /codex new <task> | continue <task> | resume <session> <task> | "
            "select <session> | status | sessions | "
            "workspace [list|set <path-or-number>|current|clear] | diff | stop | "
            "plan <task> | permissions <default|approve-for-me|read-only|full-access>",
            status="usage",
        )

    def _workspace_for(self, request: CommandRequest, task_key: str) -> str:
        selected = self._workspace_store.get(task_key)
        return selected or request.workspace or os.getcwd()

    def _workspace_command(
        self,
        request: CommandRequest,
        task_key: str,
        raw_args: str,
    ) -> CommandResult:
        args = raw_args.strip()
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else "list"
        value = parts[1].strip() if len(parts) > 1 else ""

        if action in {"current", "show"}:
            current = self._workspace_for(request, task_key)
            selected = self._workspace_store.get(task_key)
            prefix = "Selected" if selected else "Default"
            return CommandResult(f"Codex workspace\n{prefix}: {current}", status="ok")

        if action in {"clear", "reset", "unset"}:
            self._workspace_store.clear(task_key)
            return CommandResult(
                f"Codex workspace cleared. Default: {request.workspace or os.getcwd()}",
                status="ok",
            )

        if action in {"set", "use", "select"}:
            if not value:
                return CommandResult(
                    "Usage: /codex workspace set <path-or-number>",
                    status="usage",
                )
            return self._set_workspace(request, task_key, value)

        if action not in {"list", "ls"} and args:
            return self._set_workspace(request, task_key, args)

        return self._list_workspaces(request, task_key)

    def _list_workspaces(self, request: CommandRequest, task_key: str) -> CommandResult:
        roots = workspace_scan_roots(request.workspace)
        entries = discover_git_workspaces(roots)
        current = self._workspace_for(request, task_key)
        lines = ["Codex workspaces"]
        lines.append(f"Current: {current}")
        if not entries:
            lines.append("No git repositories found.")
            lines.append(
                "Set HERMES_CODEX_WORKSPACE_ROOTS with colon-separated roots "
                "to widen discovery."
            )
            return CommandResult("\n".join(lines), status="not_found")
        for index, entry in enumerate(entries[:20], start=1):
            marker = "*" if os.path.abspath(entry.path) == os.path.abspath(current) else " "
            lines.append(f"{marker} {index}. {entry.path}")
        if len(entries) > 20:
            lines.append(f"...and {len(entries) - 20} more")
        lines.append("Use `/codex workspace set <number-or-path>`.")
        return CommandResult(
            "\n".join(lines),
            status="ok",
            diagnostics={"workspaces": [entry.path for entry in entries]},
        )

    def _set_workspace(
        self,
        request: CommandRequest,
        task_key: str,
        value: str,
    ) -> CommandResult:
        workspace = value.strip()
        if workspace.isdigit():
            entries = discover_git_workspaces(workspace_scan_roots(request.workspace))
            index = int(workspace)
            if index < 1 or index > len(entries):
                return CommandResult(
                    f"Codex workspace not found: index {index}",
                    status="not_found",
                )
            workspace = entries[index - 1].path
        workspace = os.path.abspath(os.path.expanduser(workspace))
        if not os.path.isdir(workspace):
            return CommandResult(
                f"Codex workspace not found: {workspace}",
                status="not_found",
            )
        if not os.path.exists(os.path.join(workspace, ".git")):
            digest = git_digest(workspace)
            if not digest.get("available"):
                return CommandResult(
                    f"Codex workspace is not a git repository: {workspace}",
                    status="failed",
                )
            repo_root = str(digest.get("repoRoot") or workspace)
            workspace = repo_root
        selected = self._workspace_store.set(task_key, workspace)
        return CommandResult(f"Codex workspace selected:\n{selected}", status="ok")

    def _status(self, task_key: str) -> CommandResult:
        record = self._selected_or_latest_record(task_key)
        if record is None:
            return CommandResult("Codex: no task found.", status="not_found")
        return CommandResult(
            format_task_status(record),
            status=str(getattr(record, "status", "unknown") or "unknown"),
            task_id=str(getattr(record, "task_id", "") or ""),
            thread_id=str(getattr(record, "thread_id", "") or ""),
        )

    def _sessions_status(
        self,
        request: CommandRequest,
        task_key: str,
        raw_args: str = "",
    ) -> CommandResult:
        args = raw_args.strip()
        include_all = args == "all"
        workspace_query = ""
        if include_all and not self._can_show_all_sessions(request):
            return CommandResult(
                "Codex sessions all is only available in admin diagnostics mode.",
                status="forbidden",
            )
        if args.startswith("workspace "):
            workspace_query = args.split(maxsplit=1)[1].strip().lower()
        records = self._session_records_for_request(
            request,
            task_key,
            include_all=include_all,
            workspace_query=workspace_query,
            limit=50,
        )
        if not records:
            return CommandResult("Codex: no sessions found for this scope.", status="not_found")
        selected = self._selected_or_latest_record(task_key)
        selected_thread = str(getattr(selected, "thread_id", "") or "")
        lines = ["Codex sessions"]
        for index, record in enumerate(records[:10], start=1):
            marker = "*" if getattr(record, "thread_id", "") == selected_thread else " "
            title = getattr(record, "title", "") or "(untitled)"
            workspace = self._workspace_label(str(getattr(record, "workspace", "") or ""))
            thread_id = str(getattr(record, "thread_id", "") or "")
            lines.append(
                f"{index}. {marker} {getattr(record, 'task_id', '')} · "
                f"{thread_id[:8]} · {workspace} · "
                f"{getattr(record, 'status', '')} · {title}"
            )
        lines.append("Use `/codex select <number-or-id>` or `/codex resume <number-or-id> <task>`.")
        return CommandResult(
            "\n".join(lines),
            status="ok",
            diagnostics={
                "sessions": [
                    {
                        "task_id": str(getattr(record, "task_id", "") or ""),
                        "thread_id": str(getattr(record, "thread_id", "") or ""),
                        "workspace": str(getattr(record, "workspace", "") or ""),
                    }
                    for record in records[:10]
                ],
            },
        )

    def _select_session(
        self,
        request: CommandRequest,
        task_key: str,
        raw_selector: str,
    ) -> CommandResult:
        selector = raw_selector.strip()
        if not selector:
            return CommandResult("Usage: /codex select <session>", status="usage")
        record, error = self._resolve_session_selector(request, task_key, selector)
        if error is not None:
            return error
        assert record is not None
        self._selected_store.set(
            task_key,
            task_id=str(getattr(record, "task_id", "") or ""),
            thread_id=str(getattr(record, "thread_id", "") or ""),
            workspace=str(getattr(record, "workspace", "") or ""),
        )
        return CommandResult(
            "Codex session selected\n"
            f"Task: {getattr(record, 'task_id', '')}\n"
            f"Thread: {getattr(record, 'thread_id', '')}\n"
            f"Workspace: {getattr(record, 'workspace', '')}",
            status="ok",
            task_id=str(getattr(record, "task_id", "") or ""),
            thread_id=str(getattr(record, "thread_id", "") or ""),
        )

    async def _resume_session(
        self,
        request: CommandRequest,
        task_key: str,
        raw_args: str,
    ) -> CommandResult:
        parts = raw_args.strip().split(maxsplit=1)
        if len(parts) < 2:
            return CommandResult("Usage: /codex resume <session> <task>", status="usage")
        selector, prompt = parts[0], parts[1].strip()
        if not prompt:
            return CommandResult("Usage: /codex resume <session> <task>", status="usage")
        record, error = self._resolve_session_selector(request, task_key, selector)
        if error is not None:
            return error
        assert record is not None
        return await self._run(
            request,
            task_key,
            prompt,
            workspace=str(getattr(record, "workspace", "") or self._workspace_for(request, task_key)),
            new_session=False,
            plan_mode=False,
            resume_thread_id=str(getattr(record, "thread_id", "") or ""),
            select_session=True,
        )

    def _session_records_for_request(
        self,
        request: CommandRequest,
        task_key: str,
        *,
        include_all: bool = False,
        workspace_query: str = "",
        limit: int = 50,
    ) -> list[Any]:
        platform_prefix = f"{(request.platform or 'unknown').lower()}:"
        try:
            raw_records = self._registry.list(limit=limit)
        except TypeError:
            raw_records = self._registry.list()
        except Exception:
            raw_records = []
        if include_all:
            records = [
                record
                for record in raw_records
                if str(getattr(record, "task_key", "")).startswith(platform_prefix)
            ]
        else:
            records = [
                record
                for record in raw_records
                if str(getattr(record, "task_key", "") or "") == task_key
            ]
        if workspace_query:
            records = [
                record
                for record in records
                if workspace_query
                in (
                    f"{getattr(record, 'workspace', '')} "
                    f"{self._workspace_label(str(getattr(record, 'workspace', '') or ''))}"
                ).lower()
            ]
        records = [
            record
            for record in self._dedupe_session_records(records)
            if getattr(record, "thread_id", "")
        ]
        return records

    def _resolve_session_selector(
        self,
        request: CommandRequest,
        task_key: str,
        selector: str,
    ) -> tuple[Any | None, CommandResult | None]:
        selector = selector.strip()
        records = self._session_records_for_request(request, task_key, limit=100)
        if selector.isdigit():
            index = int(selector)
            if 1 <= index <= len(records):
                return records[index - 1], None
            return None, CommandResult(
                f"Codex session not found: index {index}",
                status="not_found",
            )
        lowered = selector.lower()
        matches = [
            record
            for record in records
            if str(getattr(record, "task_id", "") or "").lower().startswith(lowered)
            or str(getattr(record, "thread_id", "") or "").lower().startswith(lowered)
        ]
        if not matches:
            return None, CommandResult(
                f"Codex session not found: {selector}",
                status="not_found",
            )
        if len(matches) > 1:
            lines = ["Codex session selector is ambiguous"]
            for record in matches[:5]:
                lines.append(
                    f"- {getattr(record, 'task_id', '')} · "
                    f"{str(getattr(record, 'thread_id', '') or '')[:8]} · "
                    f"{self._workspace_label(str(getattr(record, 'workspace', '') or ''))} · "
                    f"{getattr(record, 'title', '') or '(untitled)'}"
                )
            return None, CommandResult("\n".join(lines), status="ambiguous")
        return matches[0], None

    def _selected_or_latest_record(self, task_key: str) -> Any:
        selected = self._selected_store.get(task_key)
        if selected is not None:
            record = None
            if selected.task_id:
                record = self._registry.get(task_id=selected.task_id)
            if record is None and selected.thread_id:
                record = self._session_record(task_key, selected.thread_id)
            if record is not None and getattr(record, "task_key", "") == task_key:
                return record
        return self._registry.get(task_key=task_key)

    @staticmethod
    def _can_show_all_sessions(request: CommandRequest) -> bool:
        if bool(getattr(request, "is_admin", False)):
            return True
        return os.environ.get("HERMES_CODEX_DIAGNOSTICS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _workspace_label(workspace: str) -> str:
        workspace = workspace.rstrip(os.sep)
        return os.path.basename(workspace) if workspace else "(unknown)"

    def _diff(self, task_key: str, workspace: str) -> CommandResult:
        record = self._selected_or_latest_record(task_key)
        resolved_workspace = (getattr(record, "workspace", "") if record else "") or workspace
        try:
            digest = git_digest(resolved_workspace or os.getcwd())
            return CommandResult(
                format_git_digest(digest),
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
        value = normalize_permission_profile(raw_value)
        profiles = codex_permission_profiles()
        if not raw_value.strip():
            lines = ["Codex permissions"]
            for name in ("default", "auto_review", "read_only", "full_access"):
                profile = profiles[name]
                suffix = (
                    f" / {profile['approvals_reviewer']}"
                    if profile.get("approvals_reviewer")
                    else ""
                )
                lines.append(
                    f"- {name.replace('_', '-')}: "
                    f"{profile['sandbox']} / {profile['approval_policy']}{suffix}"
                )
            lines.append("Use `/codex permissions approve-for-me` to match desktop Approve for me.")
            return CommandResult(
                "\n".join(lines),
                status="usage",
            )
        if not value or value not in profiles:
            return CommandResult(
                "Unknown Codex permission mode. Use default, approve-for-me, read-only, or full-access.",
                status="usage",
            )
        profile = profiles[value]
        sandbox = profile["sandbox"]
        approval = profile["approval_policy"]
        reviewer = profile.get("approvals_reviewer", "")
        try:
            from cli import save_config_value

            save_config_value("codex_app_server.sandbox", sandbox)
            save_config_value("codex_app_server.approval_policy", approval)
            save_config_value("codex_app_server.approvals_reviewer", reviewer)
        except Exception:
            logger.debug("Could not persist codex permission mode", exc_info=True)
        suffix = f" / {reviewer}" if reviewer else ""
        return CommandResult(
            f"Codex permissions set to {profile['label']}: {sandbox} / {approval}{suffix}.",
            status="ok",
        )

    async def _run(
        self,
        request: CommandRequest,
        task_key: str,
        prompt: str,
        *,
        workspace: str,
        new_session: bool,
        plan_mode: bool,
        resume_thread_id: str = "",
        select_session: bool = False,
    ) -> CommandResult:
        return await run_blocking(
            self._run_sync,
            request,
            task_key,
            prompt,
            workspace or os.getcwd(),
            new_session,
            plan_mode,
            resume_thread_id,
            select_session,
        )

    def _run_sync(
        self,
        request: CommandRequest,
        task_key: str,
        prompt: str,
        workspace: str,
        new_session: bool,
        plan_mode: bool,
        resume_thread_id: str = "",
        select_session: bool = False,
    ) -> CommandResult:
        codex_cfg = load_codex_cfg()
        config_overrides = codex_app_server_config_overrides(codex_cfg)
        turn_options = codex_app_server_turn_options(codex_cfg)
        prelocked_live = self._sessions.peek(task_key)
        acquired_live = None
        if prelocked_live is not None:
            if not prelocked_live.lock.acquire(blocking=False):
                return CommandResult(
                    "Codex app-server failed: a Codex task is already running for this chat. "
                    "Use `/codex stop` or wait for it to finish.",
                    status="busy",
                )
            acquired_live = prelocked_live
        live = self._sessions.get(
            task_key,
            workspace,
            new_session=new_session,
            config_overrides=config_overrides,
            resume_thread_id=resume_thread_id,
        )
        if live is None:
            if acquired_live is not None:
                acquired_live.lock.release()
            return CommandResult(
                "No live Codex app-server session found for this chat. "
                "Use `/codex new <task>` to start a fresh session.",
                status="not_found",
            )
        if acquired_live is not live:
            if acquired_live is not None:
                acquired_live.lock.release()
            if not live.lock.acquire(blocking=False):
                return CommandResult(
                    "Codex app-server failed: a Codex task is already running for this chat. "
                    "Use `/codex stop` or wait for it to finish.",
                    status="busy",
                )
            acquired_live = live
        if acquired_live is None:
            return CommandResult(
                "Codex app-server failed: a Codex task is already running for this chat. "
                "Use `/codex stop` or wait for it to finish.",
                status="busy",
            )
        bridge = CodexApprovalBridge(
            session_key=request.approval_session_key or task_key,
            notify=request.approval_notify,
        )
        approval = str(codex_cfg.get("approval_policy") or "on-request")
        sandbox = normalize_sandbox_mode(str(codex_cfg.get("sandbox") or "workspace-write"))
        approvals_reviewer = str(codex_cfg.get("approvals_reviewer") or "").strip()
        approval_callback = (
            (lambda *_args, **_kwargs: "session")
            if (
                approval == "never"
                or approvals_reviewer == "auto_review"
                or sandbox == "danger-full-access"
            )
            else bridge.callback
        )
        try:
            with bridge:
                if hasattr(live.session, "set_approval_callback"):
                    live.session.set_approval_callback(approval_callback)
                return self._run_locked(
                    live,
                    task_key,
                    prompt,
                    workspace,
                    new_session,
                    plan_mode,
                    codex_cfg,
                    turn_options,
                    request.progress_notify,
                    select_session,
                )
        finally:
            if hasattr(live.session, "set_approval_callback"):
                live.session.set_approval_callback(None)
            live.active_task_id = ""
            if acquired_live is not None:
                acquired_live.lock.release()

    def _run_locked(
        self,
        live: LiveSession,
        task_key: str,
        prompt: str,
        workspace: str,
        new_session: bool,
        plan_mode: bool,
        codex_cfg: dict[str, Any] | None = None,
        turn_options: dict[str, float] | None = None,
        progress_notify: Callable[[dict[str, Any]], None] | None = None,
        select_session: bool = False,
    ) -> CommandResult:
        codex_cfg = codex_cfg if isinstance(codex_cfg, dict) else load_codex_cfg()
        turn_options = (
            turn_options if isinstance(turn_options, dict) else codex_app_server_turn_options(codex_cfg)
        )
        model = str(codex_cfg.get("model") or read_codex_config_model())
        sandbox = normalize_sandbox_mode(str(codex_cfg.get("sandbox") or "workspace-write"))
        approval = str(codex_cfg.get("approval_policy") or "on-request")
        try:
            thread_id = live.session.ensure_started()
            live.thread_id = thread_id
        except Exception as exc:
            self._sessions.retire(task_key, live)
            return CommandResult(
                format_failure("Codex app-server failed", exc),
                status="failed",
                diagnostics={"phase": "thread/start"},
            )

        record = None if new_session else self._session_record(task_key, thread_id)
        task_id = getattr(record, "task_id", "") or task_id_for(thread_id)
        live.active_task_id = task_id
        progress_callback = self._progress_callback(
            progress_notify,
            task_id=task_id,
            thread_id=thread_id,
            workspace=workspace,
            interval_seconds=self._progress_interval_seconds(codex_cfg),
        )
        if record is None:
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
        else:
            self._registry.update(
                task_id,
                status="planning" if plan_mode else "running",
                workspace=workspace,
                thread_id=thread_id,
                turn_id="",
                model=model,
                approval_policy=approval,
                sandbox=sandbox,
                plan_mode=plan_mode,
                last_message="Codex: turn started",
            )
        if select_session:
            self._selected_store.set(
                task_key,
                task_id=task_id,
                thread_id=thread_id,
                workspace=workspace,
            )

        run_turn_kwargs = dict(turn_options)
        if progress_callback is not None:
            run_turn_kwargs["progress_callback"] = progress_callback
        turn = live.session.run_turn(user_input=prompt, **run_turn_kwargs)
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

    @staticmethod
    def _progress_callback(
        notify: Callable[[dict[str, Any]], None] | None,
        *,
        task_id: str,
        thread_id: str,
        workspace: str,
        interval_seconds: float,
    ) -> Callable[[dict[str, Any]], None] | None:
        if notify is None:
            return None
        last_emit_at = 0.0
        forced = {
            "turn_started",
            "approval_requested",
            "turn_timed_out",
        }

        def callback(event: dict[str, Any]) -> None:
            nonlocal last_emit_at
            stage = str(event.get("stage") or "")
            now = time.monotonic()
            if stage not in forced and (now - last_emit_at) < max(5.0, interval_seconds):
                return
            text = CodexCommandService._format_progress_text(
                stage,
                workspace=workspace,
                thread_id=thread_id,
                event=event,
            )
            if not text:
                return
            last_emit_at = now
            notify(
                {
                    "type": "codex_progress",
                    "task_id": task_id,
                    "thread_id": thread_id,
                    "workspace": workspace,
                    "stage": stage,
                    "text": text,
                }
            )

        return callback

    @staticmethod
    def _progress_interval_seconds(codex_cfg: dict[str, Any]) -> float:
        try:
            value = float(codex_cfg.get("progress_interval_seconds") or 60.0)
        except (TypeError, ValueError):
            return 60.0
        return value if value > 0 else 60.0

    @staticmethod
    def _format_progress_text(
        stage: str,
        *,
        workspace: str,
        thread_id: str,
        event: dict[str, Any],
    ) -> str:
        workspace_name = os.path.basename(workspace.rstrip(os.sep)) or workspace
        thread_short = thread_id[:13] if thread_id else "pending"
        if stage == "turn_started":
            return (
                "Codex is running\n"
                f"Workspace: {workspace_name} · Session: {thread_short}"
            )
        if stage == "approval_requested":
            return "Codex is waiting for command approval."
        if stage == "tool_completed":
            count = int(event.get("tool_iterations") or 0)
            suffix = f" ({count} tool step{'s' if count != 1 else ''})" if count else ""
            return f"Codex made progress{suffix}."
        if stage == "waiting":
            idle = int(float(event.get("idle_seconds") or 0))
            return f"Codex is still running. Last app-server activity was {idle}s ago."
        if stage == "turn_timed_out":
            timeout = int(float(event.get("timeout_seconds") or 0))
            return f"Codex task timed out after {timeout}s without app-server activity."
        return ""

    def _session_record(self, task_key: str, thread_id: str) -> Any:
        try:
            records = self._registry.list(task_key=task_key, limit=50)
        except TypeError:
            records = self._registry.list(limit=50)
        except Exception:
            records = []
        matches = [
            record
            for record in records
            if getattr(record, "task_key", "") == task_key
            and getattr(record, "thread_id", "") == thread_id
        ]
        if matches:
            return sorted(matches, key=lambda record: getattr(record, "started_at", 0) or 0)[0]
        return self._registry.get(task_key=task_key)

    @staticmethod
    def _dedupe_session_records(records: list[Any]) -> list[Any]:
        grouped: dict[tuple[str, str], list[Any]] = {}
        for record in records:
            key = (
                str(getattr(record, "task_key", "") or ""),
                str(getattr(record, "thread_id", "") or getattr(record, "task_id", "") or ""),
            )
            grouped.setdefault(key, []).append(record)

        result: list[Any] = []
        for group in grouped.values():
            group.sort(key=lambda record: getattr(record, "started_at", 0) or 0)
            canonical = group[0]
            latest = max(group, key=lambda record: getattr(record, "updated_at", 0) or 0)
            for attr in (
                "status",
                "turn_id",
                "completed_at",
                "last_message",
                "token_usage",
                "updated_at",
            ):
                if hasattr(canonical, attr):
                    setattr(canonical, attr, getattr(latest, attr, getattr(canonical, attr)))
            result.append(canonical)
        result.sort(key=lambda record: getattr(record, "updated_at", 0) or 0, reverse=True)
        return result

_DEFAULT_SERVICE: CodexCommandService | None = None
_DEFAULT_SERVICE_LOCK = threading.Lock()


def get_codex_command_service() -> CodexCommandService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        with _DEFAULT_SERVICE_LOCK:
            if _DEFAULT_SERVICE is None:
                _DEFAULT_SERVICE = CodexCommandService()
    return _DEFAULT_SERVICE
