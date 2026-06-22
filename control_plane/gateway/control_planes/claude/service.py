"""Platform-neutral /claude command service."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Optional

from agent.transports.claude_runtime import ClaudeRuntimeSession

from .approval import ClaudeApprovalBridge
from .doctor import format_doctor_checks, run_claude_doctor
from .event_store import ClaudeRuntimeEventStore
from .formatting import (
    format_failure,
    format_run_failure,
    format_run_success,
    format_task_status,
)
from .execution import run_blocking
from .git_digest import format_git_digest, git_digest
from .local_sessions import ClaudeLocalSession, ClaudeLocalSessionIndex
from .models import CommandRequest, CommandResult
from .narrator import ClaudeFieldNarrator, event_type_from_progress
from .registry import ClaudeTaskRegistry
from .records import make_task_record
from .runtime_config import (
    PLAN_PROMPT_PREFIX,
    DEFAULT_CLAUDE_PERMISSION_MODE,
    claude_cli_config_overrides,
    claude_runtime,
    claude_runtime_fallback,
    claude_sdk_profile_name,
    claude_cli_turn_options,
    claude_permission_profiles,
    load_claude_cfg,
    normalize_permission_mode,
    normalize_permission_profile,
    read_claude_config_model,
    resolve_claude_sdk_profile,
    safe_claude_sdk_profile_diagnostics,
)
from .selection import SelectedSessionStore
from .session_pool import ClaudeSessionPool, LiveSession
from .task_keys import build_claude_task_key, task_id_for
from .workspaces import (
    WorkspaceSelectionStore,
    discover_git_workspaces,
    workspace_scan_roots,
)

logger = logging.getLogger("gateway.run")


class ClaudeCommandService:
    """Platform-neutral implementation of the gateway /claude command."""

    def __init__(
        self,
        *,
        registry: Any = None,
        workspace_store: Any = None,
        selected_store: Any = None,
        session_factory: Optional[Callable[..., ClaudeRuntimeSession]] = None,
        event_store: Any = None,
        narrator: ClaudeFieldNarrator | None = None,
        local_session_index: ClaudeLocalSessionIndex | None = None,
    ) -> None:
        if registry is None:
            registry = ClaudeTaskRegistry()
        self._registry = registry
        self._workspace_store = workspace_store or WorkspaceSelectionStore()
        self._selected_store = selected_store or SelectedSessionStore()
        self._sessions = ClaudeSessionPool(session_factory=session_factory)
        self._event_store = event_store or ClaudeRuntimeEventStore()
        self._narrator = narrator or ClaudeFieldNarrator()
        self._local_sessions = local_session_index or ClaudeLocalSessionIndex()

    async def handle(self, request: CommandRequest) -> CommandResult:
        raw_args = (request.text or "").strip()
        parts = raw_args.split(maxsplit=1)
        subcommand = parts[0].lower() if parts else "status"
        rest = parts[1] if len(parts) > 1 else ""
        task_key = build_claude_task_key(request)

        if subcommand in {"status"}:
            return self._status(request, task_key)
        if subcommand in {"doctor", "诊断"}:
            return await self._doctor(request, task_key)
        if subcommand in {"events", "eventlog", "log"}:
            return self._events_status(request, task_key, rest)
        if subcommand in {"sessions", "session", "tasks", "ls", "list"}:
            if subcommand == "session" and rest.strip():
                return self._select_session(request, task_key, rest)
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
                "Claude 暂不支持在运行中插话；请在本轮结束后用 `/claude continue <补充要求>` 继续。",
                status="unsupported",
            )
        if subcommand in {"plan", "run", "new"}:
            prompt = rest.strip()
            if not prompt:
                return CommandResult(f"用法：/claude {subcommand} <任务>", status="usage")
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
                return CommandResult("用法：/claude continue <任务>", status="usage")
            resume_record = self._selected_or_latest_record(task_key)
            live = self._sessions.peek(task_key)
            if resume_record is None and live is None:
                return CommandResult(
                    "当前聊天还没有选中的 Claude 会话。请先用 `/claude new <任务>` 新开，"
                    "或用 `/claude resume <会话> <任务>` 接续历史会话。",
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
            "用法：/claude new <任务> | continue <任务> | resume <会话> <任务> | "
            "select <会话> | status | sessions | events | doctor | "
            "workspace [list|set <路径或序号>|current|clear] | diff | stop | "
            "plan <任务> | permissions <default|approve-for-me|read-only|full-access>",
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
            prefix = "已选择" if selected else "默认"
            return CommandResult(f"Claude 工作区\n{prefix}：{current}", status="ok")

        if action in {"clear", "reset", "unset"}:
            self._workspace_store.clear(task_key)
            return CommandResult(
                f"Claude 工作区已清除。默认工作区：{request.workspace or os.getcwd()}",
                status="ok",
            )

        if action in {"set", "use", "select"}:
            if not value:
                return CommandResult(
                    "用法：/claude workspace set <路径或序号>",
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
        lines = ["Claude 工作区"]
        lines.append(f"当前：{current}")
        if not entries:
            lines.append("没有找到 git 仓库。")
            lines.append(
                "可以设置 HERMES_CLAUDE_WORKSPACE_ROOTS，用冒号分隔多个扫描根目录。"
            )
            return CommandResult("\n".join(lines), status="not_found")
        for index, entry in enumerate(entries[:20], start=1):
            marker = "*" if os.path.abspath(entry.path) == os.path.abspath(current) else " "
            lines.append(f"{marker} {index}. {entry.path}")
        if len(entries) > 20:
            lines.append(f"...另有 {len(entries) - 20} 个")
        lines.append("使用 `/claude workspace set <序号或路径>` 选择工作区。")
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
                    f"没有找到这个工作区序号：{index}",
                    status="not_found",
                )
            workspace = entries[index - 1].path
        workspace = os.path.abspath(os.path.expanduser(workspace))
        if not os.path.isdir(workspace):
            return CommandResult(
                f"没有找到这个工作区：{workspace}",
                status="not_found",
            )
        if not os.path.exists(os.path.join(workspace, ".git")):
            digest = git_digest(workspace)
            if not digest.get("available"):
                return CommandResult(
                    f"这个工作区不是 git 仓库：{workspace}",
                    status="failed",
                )
            repo_root = str(digest.get("repoRoot") or workspace)
            workspace = repo_root
        selected = self._workspace_store.set(task_key, workspace)
        return CommandResult(f"已选择 Claude 工作区：\n{selected}", status="ok")

    def _status(self, request: CommandRequest, task_key: str) -> CommandResult:
        record = self._selected_or_latest_record(task_key)
        if record is None:
            return CommandResult("还没有 Claude 任务记录。", status="not_found")
        text = format_task_status(
            record,
            verbose=self._can_show_all_sessions(request),
        )
        try:
            events = self._event_store.tail(
                task_key=task_key,
                task_id=str(getattr(record, "task_id", "") or ""),
                thread_id=str(getattr(record, "thread_id", "") or ""),
                limit=30,
            )
            progress_text = self._narrator.status_text(
                events,
                workspace=str(getattr(record, "workspace", "") or ""),
                thread_id=str(getattr(record, "thread_id", "") or ""),
            )
            if not self._can_show_all_sessions(request):
                progress_text = _redact_status_internal_ids(progress_text, record)
            if progress_text:
                text = f"{text}\n\n现场进度\n{progress_text}"
        except Exception:
            logger.debug("Claude status event projection failed", exc_info=True)
        return CommandResult(
            text,
            status=str(getattr(record, "status", "unknown") or "unknown"),
            task_id=str(getattr(record, "task_id", "") or ""),
            thread_id=str(getattr(record, "thread_id", "") or ""),
        )

    async def _doctor(self, request: CommandRequest, task_key: str) -> CommandResult:
        record = self._selected_or_latest_record(task_key)
        workspace = (
            str(getattr(record, "workspace", "") or "")
            if record is not None
            else self._workspace_for(request, task_key)
        )
        checks = await run_blocking(
            lambda: run_claude_doctor(
                workspace=workspace,
                task_key=task_key,
                selected_record=record,
                event_store=self._event_store,
            )
        )
        status = "failed" if any(item.status == "fail" for item in checks) else "ok"
        return CommandResult(
            format_doctor_checks(checks),
            status=status,
            diagnostics={
                "checks": [
                    {"name": item.name, "status": item.status, "detail": item.detail}
                    for item in checks
                ]
            },
        )

    def _events_status(
        self,
        request: CommandRequest,
        task_key: str,
        raw_selector: str = "",
    ) -> CommandResult:
        selector = raw_selector.strip()
        if selector:
            if selector == "all" and not self._can_show_all_sessions(request):
                return CommandResult(
                    "只有管理员诊断模式可以查看所有 Claude 事件。",
                    status="forbidden",
                )
            if selector == "all":
                events = self._event_store.tail(limit=20)
                if not events:
                    return CommandResult("没有找到 Claude 运行事件。", status="not_found")
                lines = ["Claude 事件"]
                for event in events[-10:]:
                    lines.append(self._narrator.format_event_line(event))
                return CommandResult("\n".join(lines), status="ok")
            record, error = self._resolve_session_selector(request, task_key, selector)
            if error is not None:
                return error
            assert record is not None
        else:
            record = self._selected_or_latest_record(task_key)
        if record is None:
            return CommandResult("没有可查看事件的 Claude 会话。", status="not_found")
        events = self._event_store.tail(
            task_key=task_key,
            task_id=str(getattr(record, "task_id", "") or ""),
            thread_id=str(getattr(record, "thread_id", "") or ""),
            limit=20,
        )
        if not events:
            return CommandResult("没有找到 Claude 运行事件。", status="not_found")
        lines = ["Claude 事件"]
        for event in events[-10:]:
            lines.append(self._narrator.format_event_line(event))
        return CommandResult(
            "\n".join(lines),
            status="ok",
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
        verbose = args in {"verbose", "details", "ids"}
        include_all = args == "all"
        if args in {"all verbose", "all details", "all ids"}:
            include_all = True
            verbose = True
        workspace_query = ""
        if include_all and not self._can_show_all_sessions(request):
            return CommandResult(
                "只有管理员诊断模式可以查看所有 Claude 会话。",
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
            return CommandResult("当前范围内没有 Claude 会话。", status="not_found")
        selected = self._selected_or_latest_record(task_key)
        selected_thread = str(getattr(selected, "thread_id", "") or "")
        lines = ["Claude 会话"]
        for index, record in enumerate(records[:10], start=1):
            marker = "*" if getattr(record, "thread_id", "") == selected_thread else " "
            title = self._session_title(record)
            workspace = self._workspace_label(str(getattr(record, "workspace", "") or ""))
            thread_id = str(getattr(record, "thread_id", "") or "")
            prefix = f"{index}. {marker} "
            if verbose or include_all:
                prefix += f"{getattr(record, 'task_id', '')} · {thread_id[:8]} · "
            lines.append(
                f"{prefix}{workspace} · {self._session_state_label(record)} · "
                f"最近一轮：{self._status_label(getattr(record, 'status', ''))} · {title}"
            )
        lines.append("使用 `/claude select <序号>` 选择，或 `/claude resume <序号> <任务>` 接续。")
        lines.append("需要内部 ID 时用 `/claude sessions verbose`。")
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
                ]
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
            return CommandResult("用法：/claude select <会话>", status="usage")
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
            "已选择 Claude 会话\n"
            f"任务：{getattr(record, 'task_id', '')}\n"
            f"线程：{getattr(record, 'thread_id', '')}\n"
            f"工作区：{getattr(record, 'workspace', '')}",
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
            return CommandResult("用法：/claude resume <会话> <任务>", status="usage")
        selector, prompt = parts[0], parts[1].strip()
        if not prompt:
            return CommandResult("用法：/claude resume <会话> <任务>", status="usage")
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
        records.extend(
            self._local_session_records(
                request,
                task_key,
                include_all=include_all,
                workspace_query=workspace_query,
                limit=limit,
            )
        )
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
                f"没有找到这个 Claude 会话序号：{index}",
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
                f"没有找到这个 Claude 会话：{selector}",
                status="not_found",
            )
        if len(matches) > 1:
            lines = ["这个会话选择不唯一，请用更长的 ID 或序号"]
            for record in matches[:5]:
                lines.append(
                    f"- {getattr(record, 'task_id', '')} · "
                    f"{str(getattr(record, 'thread_id', '') or '')[:8]} · "
                    f"{self._workspace_label(str(getattr(record, 'workspace', '') or ''))} · "
                    f"{self._session_title(record)}"
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
        return os.environ.get("HERMES_CLAUDE_DIAGNOSTICS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _workspace_label(workspace: str) -> str:
        workspace = workspace.rstrip(os.sep)
        return os.path.basename(workspace) if workspace else "未知工作区"

    @staticmethod
    def _status_label(status: Any) -> str:
        return {
            "starting": "启动中",
            "planning": "规划中",
            "running": "运行中",
            "completed": "已完成",
            "failed": "失败",
            "interrupted": "已中断",
            "unconfirmed": "未确认",
            "recoverable_stale": "待恢复",
            "busy": "忙碌",
            "not_found": "未找到",
            "unknown": "未知",
        }.get(str(status or ""), str(status or ""))

    @staticmethod
    def _session_state_label(record: Any) -> str:
        status = str(getattr(record, "status", "") or "")
        if status in {"starting", "planning", "running", "busy"}:
            return "进行中"
        if getattr(record, "thread_id", ""):
            return "可接续"
        return "未建立会话"

    @staticmethod
    def _session_title(record: Any) -> str:
        title = " ".join(str(getattr(record, "title", "") or "").split())
        if _is_internal_plan_prompt_title(title):
            return "计划会话"
        return title or "未命名任务"

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
            return CommandResult(format_failure("Claude CLI failed", exc), status="failed")

    def _stop(self, task_key: str) -> CommandResult:
        live = self._sessions.peek(task_key)
        if live is None:
            return CommandResult(
                "停止失败：当前聊天没有正在运行的 Claude 任务。",
                status="not_found",
            )
        try:
            live.session.request_interrupt()
            if live.active_task_id:
                self._registry.update(
                    live.active_task_id,
                    status="interrupted",
                    completed_at=time.time(),
                    last_message="Claude stop requested.",
                )
            return CommandResult("已请求停止 Claude。", status="interrupted")
        except Exception as exc:
            return CommandResult(format_failure("Claude CLI failed", exc), status="failed")

    def _permissions(self, raw_value: str) -> CommandResult:
        value = normalize_permission_profile(raw_value)
        profiles = claude_permission_profiles()
        if not raw_value.strip():
            lines = ["Claude 权限模式"]
            for name in ("default", "auto_review", "read_only", "full_access"):
                profile = profiles[name]
                lines.append(
                    f"- {name.replace('_', '-')}：{profile['label']} · "
                    f"{_permission_mode_label(profile['permission_mode'])}"
                )
            lines.append(
                "影响范围：当前 Hermes `/claude` control-plane 的后续默认配置；"
                "不会回写或改写历史会话记录。"
            )
            lines.append("使用 `/claude permissions approve-for-me` 对齐桌面端“自动审批”。")
            return CommandResult(
                "\n".join(lines),
                status="usage",
            )
        if not value or value not in profiles:
            return CommandResult(
                "未知的 Claude 权限模式。可用：default、approve-for-me、read-only、full-access。",
                status="usage",
            )
        profile = profiles[value]
        permission_mode = profile["permission_mode"]
        try:
            from cli import save_config_value

            save_config_value("claude_cli.permission_mode", permission_mode)
        except Exception:
            logger.debug("Could not persist claude permission mode", exc_info=True)
        return CommandResult(
            f"Claude 权限已切换为「{profile['label']}」："
            f"{_permission_mode_label(permission_mode)}\n"
            "影响范围：后续新建或恢复的 `/claude` 运行；不会改写历史会话记录。",
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
        claude_cfg = load_claude_cfg()
        config_overrides = claude_cli_config_overrides(claude_cfg)
        turn_options = claude_cli_turn_options(claude_cfg)
        runtime = claude_runtime(claude_cfg)
        runtime_fallback = claude_runtime_fallback(claude_cfg)
        sdk_profile = claude_sdk_profile_name(claude_cfg)
        sdk_profile_config = resolve_claude_sdk_profile(claude_cfg)
        prelocked_live = self._sessions.peek(task_key)
        acquired_live = None
        if prelocked_live is not None:
            if not prelocked_live.lock.acquire(blocking=False):
                return CommandResult(
                    "当前聊天已有 Claude 任务在运行。请等待它完成，或用 `/claude stop` 停止。",
                    status="busy",
                )
            acquired_live = prelocked_live
        permission_mode = normalize_permission_mode(
            str(claude_cfg.get("permission_mode") or DEFAULT_CLAUDE_PERMISSION_MODE)
        )
        model = str(
            claude_cfg.get("model")
            or sdk_profile_config.get("model")
            or read_claude_config_model()
        )
        effort = str(claude_cfg.get("effort") or "")
        live = self._sessions.get(
            task_key,
            workspace,
            new_session=new_session,
            config_overrides=config_overrides,
            resume_thread_id=resume_thread_id,
            permission_mode=permission_mode,
            model=model,
            effort=effort,
            runtime=runtime,
            runtime_fallback=runtime_fallback,
            sdk_profile=sdk_profile,
            sdk_profile_config=sdk_profile_config,
        )
        if live is None:
            if acquired_live is not None:
                acquired_live.lock.release()
            return CommandResult(
                "当前聊天没有正在运行的 Claude 会话。请用 `/claude new <任务>` 新开一轮。",
                status="not_found",
            )
        if acquired_live is not live:
            if acquired_live is not None:
                acquired_live.lock.release()
            if not live.lock.acquire(blocking=False):
                return CommandResult(
                    "当前聊天已有 Claude 任务在运行。请等待它完成，或用 `/claude stop` 停止。",
                    status="busy",
                )
            acquired_live = live
        if acquired_live is None:
            return CommandResult(
                "当前聊天已有 Claude 任务在运行。请等待它完成，或用 `/claude stop` 停止。",
                status="busy",
            )
        bridge = ClaudeApprovalBridge(
            session_key=request.approval_session_key or task_key,
            notify=request.approval_notify,
        )
        try:
            with bridge:
                if hasattr(live.session, "set_approval_callback"):
                    live.session.set_approval_callback(bridge.callback)
                return self._run_locked(
                    live,
                    request,
                    task_key,
                    prompt,
                    workspace,
                    new_session,
                    plan_mode,
                    claude_cfg,
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
        request: CommandRequest,
        task_key: str,
        prompt: str,
        workspace: str,
        new_session: bool,
        plan_mode: bool,
        claude_cfg: dict[str, Any] | None = None,
        turn_options: dict[str, Any] | None = None,
        progress_notify: Callable[[dict[str, Any]], None] | None = None,
        select_session: bool = False,
    ) -> CommandResult:
        claude_cfg = claude_cfg if isinstance(claude_cfg, dict) else load_claude_cfg()
        runtime = claude_runtime(claude_cfg)
        sdk_profile_config = resolve_claude_sdk_profile(claude_cfg)
        turn_options = (
            turn_options if isinstance(turn_options, dict) else claude_cli_turn_options(claude_cfg)
        )
        model = str(
            claude_cfg.get("model")
            or sdk_profile_config.get("model")
            or read_claude_config_model()
        )
        permission_mode = normalize_permission_mode(
            str(claude_cfg.get("permission_mode") or DEFAULT_CLAUDE_PERMISSION_MODE)
        )
        try:
            thread_id = live.session.ensure_started()
            live.thread_id = thread_id
        except Exception as exc:
            self._sessions.retire(task_key, live)
            return CommandResult(
                format_failure("Claude CLI failed", exc),
                status="failed",
                diagnostics={"phase": "binary/start"},
            )

        record = None if new_session else self._session_record(task_key, thread_id)
        if _is_local_session_record(record):
            record = None
        task_id = getattr(record, "task_id", "") or task_id_for(thread_id)
        live.active_task_id = task_id
        turn_started_at = time.time()
        progress_callback = self._progress_callback(
            progress_notify,
            task_key=task_key,
            task_id=task_id,
            thread_id=thread_id,
            workspace=workspace,
            platform=request.platform,
            chat_id=request.chat_id,
            interval_seconds=self._progress_interval_seconds(claude_cfg),
        )
        if record is None:
            new_record = make_task_record(
                task_id=task_id,
                task_key=task_key,
                status="planning" if plan_mode else "running",
                workspace=workspace,
                thread_id=thread_id,
                turn_id="",
                model=model,
                permission_mode=permission_mode,
                prompt=prompt,
                last_message="Claude: turn started",
            )
            new_record.turn_started_at = turn_started_at
            self._registry.upsert(new_record)
        else:
            self._registry.update(
                task_id,
                status="planning" if plan_mode else "running",
                workspace=workspace,
                thread_id=thread_id,
                turn_id="",
                model=model,
                permission_mode=permission_mode,
                title=" ".join(prompt.strip().split())[:80],
                turn_started_at=turn_started_at,
                completed_at=None,
                token_usage={},
                last_message="Claude: turn started",
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
        try:
            turn = live.session.run_turn(user_input=prompt, **run_turn_kwargs)
        except Exception as exc:
            turn = SimpleNamespace(
                final_text="",
                error=str(exc),
                interrupted=False,
                should_retire=True,
                turn_id="",
                session_id=thread_id,
                token_usage_total={},
            )
        status = "completed"
        last_message = "Claude: turn completed"
        if getattr(turn, "error", None):
            status = "failed"
            last_message = str(turn.error)
        elif getattr(turn, "interrupted", False):
            status = "interrupted"
            last_message = "Claude: turn interrupted"
        elif getattr(turn, "warning", None):
            last_message = str(turn.warning)
        resolved_thread_id = (
            str(
                getattr(turn, "session_id", "")
                or getattr(turn, "thread_id", "")
                or getattr(live.session, "session_id", "")
                or getattr(live.session, "thread_id", "")
                or thread_id
            )
        )
        if resolved_thread_id:
            thread_id = resolved_thread_id
            live.thread_id = resolved_thread_id

        self._registry.update(
            task_id,
            status=status,
            thread_id=thread_id,
            turn_id=getattr(turn, "turn_id", None) or "",
            completed_at=time.time(),
            last_message=last_message,
            token_usage=getattr(turn, "token_usage_total", None) or {},
        )
        if select_session:
            self._selected_store.set(
                task_key,
                task_id=task_id,
                thread_id=thread_id,
                workspace=workspace,
            )
        self._append_runtime_event(
            task_key=task_key,
            task_id=task_id,
            thread_id=thread_id,
            turn_id=getattr(turn, "turn_id", None) or "",
            platform=request.platform,
            chat_id=request.chat_id,
            event_type=(
                "turn.failed" if getattr(turn, "error", None)
                else "turn.interrupted" if getattr(turn, "interrupted", False)
                else "turn.completed"
            ),
            payload={
                "status": status,
                "error": str(getattr(turn, "error", "") or ""),
                "warning": str(getattr(turn, "warning", "") or ""),
                "error_kind": str(getattr(turn, "error_kind", "") or ""),
                "exit_status": getattr(turn, "exit_status", None),
                "api_retry_count": getattr(turn, "api_retry_count", 0),
                "raw_tail": getattr(turn, "raw_output_tail", []) or [],
                "runtime": str(getattr(turn, "runtime", "") or runtime),
                "runtime_profile": str(
                    getattr(turn, "runtime_profile", "")
                    or sdk_profile_config.get("name")
                    or ""
                ),
                "fallback_runtime": str(getattr(turn, "fallback_runtime", "") or ""),
                "fallback_reason": str(getattr(turn, "fallback_reason", "") or ""),
                "interrupted": bool(getattr(turn, "interrupted", False)),
            },
            notify=progress_notify,
            workspace=workspace,
        )
        if getattr(turn, "fallback_reason", None):
            self._append_runtime_event(
                task_key=task_key,
                task_id=task_id,
                thread_id=thread_id,
                turn_id=getattr(turn, "turn_id", None) or "",
                platform=request.platform,
                chat_id=request.chat_id,
                event_type="runtime.fallback",
                payload={
                    "from": runtime,
                    "to": str(getattr(turn, "fallback_runtime", "") or ""),
                    "reason": str(getattr(turn, "fallback_reason", "") or ""),
                    "sdk_profile": safe_claude_sdk_profile_diagnostics(sdk_profile_config),
                },
                notify=progress_notify,
                workspace=workspace,
            )
        if getattr(turn, "should_retire", False):
            self._sessions.retire(task_key, live)

        text = getattr(turn, "final_text", "") or ""
        output = (
            format_run_failure(workspace, thread_id, str(turn.error))
            if getattr(turn, "error", None)
            else format_run_success(workspace, thread_id, text)
        )
        if not getattr(turn, "error", None) and getattr(turn, "warning", None):
            output = f"{output}\n\n提醒：{format_failure('Claude CLI', str(turn.warning))}"
        return CommandResult(
            output,
            status=status,
            task_id=task_id,
            thread_id=thread_id,
            diagnostics={
                "turn_id": getattr(turn, "turn_id", None) or "",
                "warning": str(getattr(turn, "warning", "") or ""),
                "error_kind": str(getattr(turn, "error_kind", "") or ""),
                "runtime": str(getattr(turn, "runtime", "") or runtime),
                "runtime_profile": str(
                    getattr(turn, "runtime_profile", "")
                    or sdk_profile_config.get("name")
                    or ""
                ),
                "fallback_runtime": str(getattr(turn, "fallback_runtime", "") or ""),
            },
        )

    def _progress_callback(
        self,
        notify: Callable[[dict[str, Any]], None] | None,
        *,
        task_key: str,
        task_id: str,
        thread_id: str,
        workspace: str,
        platform: str,
        chat_id: str,
        interval_seconds: float,
    ) -> Callable[[dict[str, Any]], None] | None:
        last_emit_at = 0.0
        last_by_key: dict[str, float] = {}
        dedupe_seconds = max(5.0, interval_seconds)

        def callback(event: dict[str, Any]) -> None:
            nonlocal last_emit_at
            event_type = event_type_from_progress(event)
            stored = self._append_runtime_event(
                task_key=task_key,
                task_id=task_id,
                thread_id=thread_id,
                turn_id="",
                platform=platform,
                chat_id=chat_id,
                event_type=event_type,
                payload=event,
                notify=None,
                workspace=workspace,
            )
            if notify is None or stored is None:
                return
            try:
                recent = self._event_store.tail(task_key=task_key, task_id=task_id, limit=30)
                narration = self._narrator.narrate(
                    stored,
                    recent_events=recent,
                    workspace=workspace,
                    thread_id=thread_id,
                )
            except Exception:
                logger.debug("Claude progress narration failed", exc_info=True)
                return
            if narration is None:
                return
            now = time.monotonic()
            dedupe_key = narration.dedupe_key or event_type
            if (now - last_by_key.get(dedupe_key, 0.0)) < 2.0:
                return
            if (
                not narration.force
                and (now - last_emit_at) < dedupe_seconds
                and (now - last_by_key.get(dedupe_key, 0.0)) < dedupe_seconds
            ):
                return
            last_emit_at = now
            last_by_key[dedupe_key] = now
            notify(
                {
                    "type": "claude_progress",
                    "task_id": task_id,
                    "thread_id": thread_id,
                    "workspace": workspace,
                    "stage": event.get("stage") or event_type,
                    "event_type": event_type,
                    "importance": narration.importance,
                    "dedupe_key": dedupe_key,
                    "evidence": list(narration.evidence),
                    "text": narration.render(),
                }
            )

        return callback

    def _append_runtime_event(
        self,
        *,
        task_key: str,
        task_id: str,
        thread_id: str,
        turn_id: str = "",
        platform: str = "",
        chat_id: str = "",
        event_type: str,
        payload: dict[str, Any],
        notify: Callable[[dict[str, Any]], None] | None = None,
        workspace: str = "",
    ):
        try:
            stored = self._event_store.append(
                task_key=task_key,
                task_id=task_id,
                thread_id=thread_id,
                turn_id=turn_id,
                platform=platform,
                chat_id=chat_id,
                event_type=event_type,
                payload=payload,
            )
        except Exception:
            logger.debug("Claude runtime event append failed", exc_info=True)
            return None
        if notify is not None:
            try:
                recent = self._event_store.tail(task_key=task_key, task_id=task_id, limit=30)
                narration = self._narrator.narrate(
                    stored,
                    recent_events=recent,
                    workspace=workspace,
                    thread_id=thread_id,
                )
                if narration is not None:
                    notify(
                        {
                            "type": "claude_progress",
                            "task_id": task_id,
                            "thread_id": thread_id,
                            "workspace": workspace,
                            "stage": event_type,
                            "event_type": event_type,
                            "importance": narration.importance,
                            "dedupe_key": narration.dedupe_key or event_type,
                            "evidence": list(narration.evidence),
                            "text": narration.render(),
                        }
                    )
            except Exception:
                logger.debug("Claude terminal narration failed", exc_info=True)
        return stored

    @staticmethod
    def _progress_interval_seconds(claude_cfg: dict[str, Any]) -> float:
        try:
            value = float(claude_cfg.get("progress_interval_seconds") or 60.0)
        except (TypeError, ValueError):
            return 60.0
        return value if value > 0 else 60.0

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
        local = self._local_session_record(task_key, thread_id)
        if local is not None:
            return local
        return self._registry.get(task_key=task_key)

    def _local_session_records(
        self,
        request: CommandRequest,
        task_key: str,
        *,
        include_all: bool,
        workspace_query: str,
        limit: int,
    ) -> list[Any]:
        cwd = "" if include_all or workspace_query else self._workspace_for(request, task_key)
        try:
            sessions = self._local_sessions.list_recent(limit=limit, cwd=cwd)
        except Exception:
            logger.debug("Claude local session scan failed", exc_info=True)
            return []
        records = [
            self._record_from_local_session(task_key, session)
            for session in sessions
            if session.session_id
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
        return records

    def _local_session_record(self, task_key: str, thread_id: str) -> Any | None:
        try:
            session = self._local_sessions.get(thread_id)
        except Exception:
            logger.debug("Claude local session lookup failed", exc_info=True)
            return None
        if session is None:
            return None
        return self._record_from_local_session(task_key, session)

    @staticmethod
    def _record_from_local_session(task_key: str, session: ClaudeLocalSession) -> Any:
        updated_at = _timestamp_from_iso(session.updated_at)
        started_at = _timestamp_from_iso(session.created_at) or updated_at
        return SimpleNamespace(
            task_id=f"local:{session.session_id}",
            task_key=task_key,
            status="completed",
            workspace=session.cwd,
            thread_id=session.session_id,
            turn_id="",
            model="",
            permission_mode=session.permission_mode or DEFAULT_CLAUDE_PERMISSION_MODE,
            title=session.title,
            started_at=started_at or time.time(),
            updated_at=updated_at or time.time(),
            completed_at=updated_at or None,
            last_message="Claude 本地会话，可接续",
            token_usage={},
            recent_events=[],
            source="claude_local",
            source_path=session.source_path,
        )

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


def _permission_mode_label(value: Any) -> str:
    raw = str(value or "")
    return {
        "acceptEdits": "允许编辑",
        "auto": "自动模式",
        "plan": "只规划",
        "default": "默认确认",
        "dontAsk": "不询问",
        "bypassPermissions": "跳过权限检查",
    }.get(raw, raw)


def _is_internal_plan_prompt_title(title: str) -> bool:
    return title.startswith("Create a detailed implementation plan first.")


def _redact_status_internal_ids(text: str, record: Any) -> str:
    redacted = text
    for value in (
        getattr(record, "task_id", ""),
        getattr(record, "thread_id", ""),
    ):
        value = str(value or "")
        if value:
            redacted = redacted.replace(value, "当前会话")
            redacted = redacted.replace(value[:8], "当前会话")
    return redacted


def _is_local_session_record(record: Any) -> bool:
    return bool(record is not None and getattr(record, "source", "") == "claude_local")


def _timestamp_from_iso(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return 0.0


_DEFAULT_SERVICE: ClaudeCommandService | None = None
_DEFAULT_SERVICE_LOCK = threading.Lock()


def get_claude_command_service() -> ClaudeCommandService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        with _DEFAULT_SERVICE_LOCK:
            if _DEFAULT_SERVICE is None:
                _DEFAULT_SERVICE = ClaudeCommandService()
    return _DEFAULT_SERVICE
