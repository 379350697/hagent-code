import json
from types import SimpleNamespace
import time

import pytest

from gateway.control_planes.claude import (
    ClaudeCommandService,
    CommandRequest,
    build_claude_task_key,
)
from gateway.control_planes.claude.event_store import (
    ClaudeRuntimeEvent,
    ClaudeRuntimeEventStore,
)
from gateway.control_planes.claude.formatting import format_failure
from gateway.control_planes.claude.narrator import ClaudeFieldNarrator
from gateway.control_planes.claude.doctor import DoctorCheck, format_doctor_checks
from gateway.control_planes.claude.records import make_task_record


class FakeClaudeSession:
    def __init__(
        self,
        *,
        cwd,
        approval_callback=None,
        config_overrides=None,
        resume_thread_id="",
        permission_mode="acceptEdits",
        model="",
        effort="",
        runtime="",
        runtime_fallback="",
        sdk_profile_config=None,
        **kwargs,
    ):
        del kwargs
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.config_overrides = list(config_overrides or [])
        self.resume_thread_id = resume_thread_id
        self.permission_mode = permission_mode
        self.model = model
        self.effort = effort
        self.runtime = runtime
        self.runtime_fallback = runtime_fallback
        self.sdk_profile_config = dict(sdk_profile_config or {})
        self.thread_id = resume_thread_id or f"claude-thread-{cwd.rsplit('/', 1)[-1] or 'root'}"
        self.session_id = self.thread_id
        self.interrupted = False
        self.closed = False
        self.turns = 0
        self.inputs = []
        self.run_turn_options = []

    def ensure_started(self):
        return self.thread_id

    def run_turn(self, user_input, **options):
        self.turns += 1
        self.inputs.append(user_input)
        self.run_turn_options.append(dict(options))
        return SimpleNamespace(
            final_text=f"ok: {user_input}",
            error=None,
            interrupted=False,
            should_retire=False,
            turn_id=f"turn-{self.turns}",
            session_id=self.thread_id,
            token_usage_total={},
        )

    def request_interrupt(self):
        self.interrupted = True

    def close(self):
        self.closed = True

    def set_approval_callback(self, callback):
        self.approval_callback = callback


class RealisticNewClaudeSession(FakeClaudeSession):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thread_id = self.resume_thread_id
        self.session_id = self.resume_thread_id

    def ensure_started(self):
        return self.thread_id

    def run_turn(self, user_input, **options):
        self.turns += 1
        self.inputs.append(user_input)
        self.run_turn_options.append(dict(options))
        if not self.session_id:
            self.session_id = "real-claude-session-1"
            self.thread_id = self.session_id
        return SimpleNamespace(
            final_text=f"ok: {user_input}",
            error=None,
            interrupted=False,
            should_retire=False,
            turn_id=f"turn-{self.turns}",
            session_id=self.session_id,
            token_usage_total={},
        )


class TimeoutClaudeSession(FakeClaudeSession):
    def ensure_started(self):
        raise FileNotFoundError("Claude binary not found")


class FailedTurnClaudeSession(FakeClaudeSession):
    def run_turn(self, user_input, **options):
        self.turns += 1
        self.inputs.append(user_input)
        self.run_turn_options.append(dict(options))
        return SimpleNamespace(
            final_text="",
            error="Claude turn timed out after 600.0s (hard)",
            interrupted=False,
            should_retire=True,
            turn_id=f"turn-{self.turns}",
            session_id=self.thread_id,
            token_usage_total={},
        )


class WarningTurnClaudeSession(FakeClaudeSession):
    def run_turn(self, user_input, **options):
        self.turns += 1
        self.inputs.append(user_input)
        self.run_turn_options.append(dict(options))
        return SimpleNamespace(
            final_text="gold is 2300",
            error=None,
            warning="Claude returned a successful result event, but the CLI process exited with status 1.",
            error_kind="success_result_then_exit_1",
            exit_status=1,
            api_retry_count=3,
            raw_output_tail=["{\"type\":\"system\",\"subtype\":\"api_retry\"}"],
            interrupted=False,
            should_retire=False,
            turn_id=f"turn-{self.turns}",
            session_id=self.thread_id,
            token_usage_total={},
        )


class ApprovalClaudeSession(FakeClaudeSession):
    last_instance = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ApprovalClaudeSession.last_instance = self

    def run_turn(self, user_input, **options):
        if self.approval_callback is not None:
            try:
                self.approval_callback("touch file.txt", "Claude requests exec in /repo")
            except Exception as exc:
                self.turns += 1
                self.inputs.append(user_input)
                self.run_turn_options.append(dict(options))
                return SimpleNamespace(
                    final_text="",
                    error=str(exc),
                    interrupted=False,
                    should_retire=False,
                    turn_id=f"turn-{self.turns}",
                    session_id=self.thread_id,
                    token_usage_total={},
                )
        return super().run_turn(user_input, **options)


class ProgressClaudeSession(FakeClaudeSession):
    def run_turn(self, user_input, **options):
        callback = options.get("progress_callback")
        if callback is not None:
            callback({"stage": "turn_started"})
            callback({"stage": "notification", "method": "item/started",
                      "item": {"type": "tool_use", "name": "Bash"}})
            callback({"stage": "notification", "method": "item/completed",
                      "item": {"type": "tool_use", "name": "Bash"}})
            callback({"stage": "tool_completed", "tool_iterations": 1})
        return super().run_turn(user_input, **options)


class MemoryRegistry:
    def __init__(self):
        self.records = {}
        self.latest_by_key = {}

    def upsert(self, record):
        self.records[record.task_id] = record
        self.latest_by_key[record.task_key] = record.task_id
        return record

    def update(self, task_id, **fields):
        record = self.records.get(task_id)
        if record is None:
            return None
        for key, value in fields.items():
            if hasattr(record, key):
                setattr(record, key, value)
        record.updated_at = time.time()
        return record

    def get(self, task_id=None, *, task_key=None, thread_id=None):
        if task_id:
            return self.records.get(task_id)
        if task_key:
            return self.records.get(self.latest_by_key.get(task_key, ""))
        if thread_id:
            for record in self.records.values():
                if record.thread_id == thread_id:
                    return record
        return None

    def list(self, *, task_key=None, limit=10):
        records = list(self.records.values())
        if task_key:
            records = [record for record in records if record.task_key == task_key]
        return records[:limit]


class MemoryWorkspaceStore:
    def __init__(self):
        self.values = {}

    def get(self, task_key):
        return self.values.get(task_key, "")

    def set(self, task_key, workspace):
        self.values[task_key] = workspace
        return workspace

    def clear(self, task_key):
        self.values.pop(task_key, None)


class MemorySelectedStore:
    def __init__(self):
        self.values = {}

    def get(self, task_key):
        return self.values.get(task_key)

    def set(self, task_key, *, task_id, thread_id, workspace=""):
        from gateway.control_planes.claude.selection import SelectedSession

        selected = SelectedSession(
            task_id=task_id,
            thread_id=thread_id,
            workspace=workspace,
            selected_at=1.0,
        )
        self.values[task_key] = selected
        return selected

    def clear(self, task_key):
        self.values.pop(task_key, None)


def _service(
    tmp_path,
    monkeypatch,
    session_factory=FakeClaudeSession,
    *,
    registry=None,
    selected_store=None,
    event_store=None,
    local_session_index=None,
):
    from gateway.control_planes.claude import service as service_mod

    monkeypatch.setattr(service_mod, "load_claude_cfg", lambda: {})
    monkeypatch.setattr(service_mod, "read_claude_config_model", lambda: "claude-sonnet-4.5")
    return ClaudeCommandService(
        registry=registry or MemoryRegistry(),
        workspace_store=MemoryWorkspaceStore(),
        selected_store=selected_store or MemorySelectedStore(),
        session_factory=session_factory,
        event_store=event_store or ClaudeRuntimeEventStore(str(tmp_path / "events.sqlite3")),
        local_session_index=local_session_index,
    )


def _make_git_repo(path):
    git_dir = path / ".git"
    git_dir.mkdir(parents=True)
    (path / "README.md").write_text("test\n", encoding="utf-8")
    return path


def test_claude_task_key_is_platform_scoped() -> None:
    discord = CommandRequest(platform="discord", chat_id="123", thread_id="", text="")
    telegram = CommandRequest(platform="telegram", chat_id="123", thread_id="", text="")

    assert build_claude_task_key(discord) == "discord:123:main"
    assert build_claude_task_key(telegram) == "telegram:123:main"
    assert build_claude_task_key(discord) != build_claude_task_key(telegram)


@pytest.mark.asyncio
async def test_claude_status_isolated_by_platform_and_chat(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)

    discord_req = CommandRequest(
        platform="discord",
        chat_id="42",
        text="new fix discord thing",
        workspace="/repo-discord",
    )
    telegram_req = CommandRequest(
        platform="telegram",
        chat_id="42",
        text="new fix telegram thing",
        workspace="/repo-telegram",
    )

    await service.handle(discord_req)
    await service.handle(telegram_req)

    discord_status = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="status")
    )
    telegram_status = await service.handle(
        CommandRequest(platform="telegram", chat_id="42", text="status")
    )

    assert "/repo-discord" in discord_status.text
    assert "/repo-telegram" in telegram_status.text
    assert "/repo-telegram" not in discord_status.text
    assert "/repo-discord" not in telegram_status.text


@pytest.mark.asyncio
async def test_claude_continue_requires_live_session(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)

    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="continue more")
    )

    assert result.status == "not_found"
    assert "/claude new" in result.text


@pytest.mark.asyncio
async def test_claude_continue_updates_existing_session_record(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)

    first = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new 123", workspace="/repo")
    )
    second = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="continue 456", workspace="/repo")
    )
    status = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="status", workspace="/repo")
    )
    sessions = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="sessions", workspace="/repo")
    )

    assert first.thread_id == second.thread_id
    assert first.task_id == second.task_id
    assert len(service._registry.records) == 1
    record = service._registry.get(task_id=first.task_id)
    assert record is not None
    assert record.title == "123"
    assert "任务：123" in status.text
    assert "任务：456" not in status.text
    assert "轮次：turn-2" in status.text
    assert first.task_id not in status.text
    assert first.thread_id not in status.text
    assert "123" in sessions.text
    assert "456" not in sessions.text
    assert first.thread_id[:8] not in sessions.text


@pytest.mark.asyncio
async def test_claude_status_reports_no_record(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="status")
    )
    assert result.status == "not_found"
    assert "还没有" in result.text


@pytest.mark.asyncio
async def test_claude_new_session_creates_record_and_progress_events(
    tmp_path, monkeypatch
) -> None:
    service = _service(tmp_path, monkeypatch, ProgressClaudeSession)
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new build feature", workspace="/repo")
    )
    assert result.status == "completed"
    assert result.thread_id
    events = service._event_store.tail(
        task_key="discord:42:main",
        task_id=result.task_id,
        limit=20,
    )
    assert any(event.event_type == "turn.completed" for event in events)
    assert any(event.event_type == "progress.tool_completed" for event in events)


@pytest.mark.asyncio
async def test_claude_new_defaults_to_sdk_runtime_profile(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENCODEGO_API_KEY", "sk-test")
    service = _service(tmp_path, monkeypatch)
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new sdk task", workspace="/repo")
    )
    live = service._sessions.peek("discord:42:main")

    assert result.status == "completed"
    assert live is not None
    assert live.runtime == "agent_sdk"
    assert live.runtime_fallback == "cli"
    assert live.sdk_profile == "opencodego"
    assert live.session.runtime == "agent_sdk"
    assert live.session.sdk_profile_config["name"] == "opencodego"


@pytest.mark.asyncio
async def test_claude_new_session_persists_real_session_id_after_first_turn(
    tmp_path, monkeypatch
) -> None:
    selected_store = MemorySelectedStore()
    service = _service(
        tmp_path,
        monkeypatch,
        RealisticNewClaudeSession,
        selected_store=selected_store,
    )

    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new bind session", workspace="/repo")
    )

    record = service._registry.get(task_key="discord:42:main")
    selected = selected_store.get("discord:42:main")
    assert result.thread_id == "real-claude-session-1"
    assert record is not None
    assert record.thread_id == "real-claude-session-1"
    assert selected is not None
    assert selected.thread_id == "real-claude-session-1"


@pytest.mark.asyncio
async def test_claude_failed_turn_marks_record_failed(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch, FailedTurnClaudeSession)
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new bad task", workspace="/repo")
    )
    assert result.status == "failed"
    record = service._registry.get(task_key="discord:42:main")
    assert record is not None
    assert record.status == "failed"
    assert "timed out" in record.last_message.lower() or "超时" in record.last_message


@pytest.mark.asyncio
async def test_claude_success_result_then_exit_warning_stays_completed(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch, WarningTurnClaudeSession)
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new gold price", workspace="/repo")
    )
    assert result.status == "completed"
    assert "gold is 2300" in result.text
    assert "提醒" in result.text
    assert result.diagnostics["error_kind"] == "success_result_then_exit_1"
    events = service._event_store.tail(
        task_key="discord:42:main",
        task_id=result.task_id,
        limit=10,
    )
    terminal = next(event for event in events if event.event_type == "turn.completed")
    assert terminal.payload["error_kind"] == "success_result_then_exit_1"
    assert terminal.payload["api_retry_count"] == 3


@pytest.mark.asyncio
async def test_claude_binary_not_found_returns_failed(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch, TimeoutClaudeSession)
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new fail", workspace="/repo")
    )
    assert result.status == "failed"
    assert "Claude" in result.text or "claude" in result.text.lower()


@pytest.mark.asyncio
async def test_claude_runtime_events_are_persisted_and_queryable(
    tmp_path, monkeypatch
) -> None:
    service = _service(tmp_path, monkeypatch)
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new task", workspace="/repo")
    )
    events = service._event_store.tail(
        task_key="discord:42:main",
        task_id=result.task_id,
        limit=10,
    )
    assert events
    assert any(event.event_type == "turn.completed" for event in events)


@pytest.mark.asyncio
async def test_claude_startup_sweep_interrupts_orphan_running_record(
    tmp_path, monkeypatch
) -> None:
    service = _service(tmp_path, monkeypatch)
    record = make_task_record(
        task_id="task-orphan",
        task_key="discord:42:main",
        status="running",
        workspace="/repo",
        thread_id="claude-thread-orphan",
        turn_id="",
        model="glm-5.2",
        permission_mode="bypassPermissions",
        prompt="deploy task",
        last_message="Claude: turn started",
    )
    service._registry.upsert(record)

    changed = service.sweep_stale_tasks(startup=True)
    record = service._registry.get(task_id="task-orphan")
    events = service._event_store.tail(
        task_key="discord:42:main",
        task_id="task-orphan",
        limit=10,
    )

    assert changed == 1
    assert record is not None
    assert record.status == "interrupted"
    assert "gateway restart" in record.last_message
    interrupted = next(event for event in events if event.event_type == "turn.interrupted")
    assert interrupted.payload["error_kind"] == "gateway_startup_without_live_turn"


@pytest.mark.asyncio
async def test_claude_status_marks_stale_running_record_interrupted(
    tmp_path, monkeypatch
) -> None:
    from gateway.control_planes.claude import service as service_mod

    service = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(service_mod, "load_claude_cfg", lambda: {"stale_running_seconds": 1})
    record = make_task_record(
        task_id="task-stale",
        task_key="discord:42:main",
        status="running",
        workspace="/repo",
        thread_id="claude-thread-stale",
        turn_id="",
        model="glm-5.2",
        permission_mode="bypassPermissions",
        prompt="deploy task",
        last_message="Claude: turn started",
    )
    service._registry.upsert(record)
    service._registry.records["task-stale"].updated_at = time.time() - 5

    status = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="status", workspace="/repo")
    )
    record = service._registry.get(task_id="task-stale")

    assert status.status == "interrupted"
    assert record is not None
    assert record.status == "interrupted"


@pytest.mark.asyncio
async def test_claude_status_recovers_terminal_event_for_running_record(
    tmp_path, monkeypatch
) -> None:
    service = _service(tmp_path, monkeypatch)
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new deploy task", workspace="/repo")
    )
    service._registry.update(
        result.task_id,
        status="running",
        completed_at=None,
        last_message="Claude: turn started",
    )
    service._event_store.append(
        task_key="discord:42:main",
        task_id=result.task_id,
        thread_id=result.thread_id,
        turn_id="turn-terminal",
        event_type="turn.completed",
        payload={"status": "completed"},
    )

    status = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="status", workspace="/repo")
    )
    record = service._registry.get(task_id=result.task_id)

    assert status.status == "completed"
    assert record is not None
    assert record.status == "completed"
    assert record.turn_id == "turn-terminal"


@pytest.mark.asyncio
async def test_claude_runtime_event_payload_is_redacted(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new task", workspace="/repo")
    )
    service._event_store.append(
        task_key="discord:42:main",
        task_id=result.task_id,
        thread_id=result.thread_id,
        event_type="progress.notification",
        payload={
            "api_key": "sk-super-secret-key",
            "authorization": "Bearer abc123",
            "safe_field": "visible",
        },
    )
    events = service._event_store.tail(
        task_key="discord:42:main",
        task_id=result.task_id,
        limit=10,
    )
    redacted_event = next(event for event in events if event.event_type == "progress.notification")
    assert redacted_event.payload["api_key"] == "[REDACTED]"
    assert redacted_event.payload["authorization"] == "[REDACTED]"
    assert redacted_event.payload["safe_field"] == "visible"


@pytest.mark.asyncio
async def test_claude_permissions_match_expected_profiles(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="permissions")
    )
    assert result.status == "usage"
    assert "default" in result.text
    assert "approve-for-me" in result.text
    assert "read-only" in result.text
    assert "full-access" in result.text


@pytest.mark.asyncio
async def test_claude_permissions_full_access_profile(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="permissions full-access")
    )
    assert result.status == "ok"
    assert "完全访问" in result.text


@pytest.mark.asyncio
async def test_claude_workspace_list_and_selection_drive_new_session(
    tmp_path, monkeypatch
) -> None:
    service = _service(tmp_path, monkeypatch)
    repo_a = _make_git_repo(tmp_path / "repoA")
    repo_b = _make_git_repo(tmp_path / "repoB")
    monkeypatch.setenv(
        "HERMES_CLAUDE_WORKSPACE_ROOTS",
        f"{tmp_path}",
    )

    listing = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="workspace list",
            workspace=str(tmp_path),
        )
    )
    assert listing.status == "ok"
    assert "repoA" in listing.text
    assert "repoB" in listing.text

    select = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text=f"workspace set 1",
            workspace=str(tmp_path),
        )
    )
    assert select.status == "ok"
    assert "repoA" in select.text


@pytest.mark.asyncio
async def test_claude_workspace_selection_is_platform_scoped(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    repo_a = _make_git_repo(tmp_path / "repoA")
    repo_b = _make_git_repo(tmp_path / "repoB")
    monkeypatch.setenv(
        "HERMES_CLAUDE_WORKSPACE_ROOTS",
        f"{tmp_path}",
    )
    await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="workspace set 1",
            workspace=str(tmp_path),
        )
    )
    current = await service.handle(
        CommandRequest(
            platform="telegram",
            chat_id="42",
            text="workspace current",
            workspace=str(tmp_path),
        )
    )
    # Telegram chat should not inherit the Discord chat's workspace choice.
    assert "repoA" not in current.text


@pytest.mark.asyncio
async def test_claude_stop_without_live_session_is_not_found(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="stop")
    )
    assert result.status == "not_found"


@pytest.mark.asyncio
async def test_claude_doctor_runs_checks(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    repo = _make_git_repo(tmp_path / "repo")
    monkeypatch.setenv("HERMES_CLAUDE_WORKSPACE_ROOTS", str(tmp_path))
    monkeypatch.setenv("HERMES_CLAUDE_DOCTOR_SMOKE", "0")
    result = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="doctor",
            workspace=str(repo),
        )
    )
    # Claude CLI may or may not be installed in test env, but doctor should run.
    assert "Claude 诊断" in result.text
    assert "工作区" in result.text


def test_claude_doctor_uses_shared_binary_resolver(tmp_path, monkeypatch) -> None:
    from gateway.control_planes.claude import doctor as doctor_mod

    monkeypatch.setenv("HERMES_CLAUDE_DOCTOR_SMOKE", "0")
    fake_claude = tmp_path / "claude"
    fake_claude.write_text("#!/usr/bin/env sh\necho '2.1.185 (Claude Code)'\n")
    fake_claude.chmod(0o755)
    monkeypatch.setattr(
        doctor_mod,
        "resolve_claude_binary",
        lambda configured="auto": str(fake_claude),
    )

    checks = doctor_mod.run_claude_doctor(
        workspace=str(tmp_path),
        task_key="discord:42:main",
    )

    cli = next(check for check in checks if check.name == "Claude CLI")
    version = next(check for check in checks if check.name == "Claude 版本")
    assert cli.status == "pass"
    assert cli.detail == str(fake_claude)
    assert version.status == "pass"
    assert "Claude Code" in version.detail


def test_claude_doctor_format_hides_internal_details() -> None:
    text = format_doctor_checks([
        DoctorCheck(
            "Claude SDK",
            "pass",
            "profile=opencodego; base_url=http://127.0.0.1:15721/; "
            "model=glm-5.2; key_source=~/.claude/settings.json",
        ),
        DoctorCheck(
            "Claude CLI",
            "pass",
            "/home/wl/.vscode/extensions/anthropic.claude-code/resources/native-binary/claude",
        ),
        DoctorCheck("Claude 版本", "pass", "2.1.185 (Claude Code)"),
        DoctorCheck(
            "运行配置",
            "pass",
            "runtime=agent_sdk; fallback=cli; profile=opencodego; "
            "model=glm-5.2; permission_mode=bypassPermissions; "
            "turn_timeout=1800s; idle_timeout=600s",
        ),
        DoctorCheck("Claude SDK smoke test", "pass", "OK"),
        DoctorCheck("工作区", "pass", "/home/wl/projects/hagent-code"),
        DoctorCheck("事件库", "pass", "/home/wl/.hermes/claude-control-plane/events.sqlite3; 最近事件 1 条"),
        DoctorCheck("当前会话", "pass", "8388ca05 · /home/wl/projects/hagent-code"),
        DoctorCheck("会话隔离键", "pass", "discord:1478651348596818023:main"),
    ])

    assert "Claude 可用性" in text
    assert "SDK 已连接" in text
    assert "运行模式" in text
    assert "短任务测试" in text
    assert "事件库" not in text
    assert "会话隔离键" not in text
    assert "base_url" not in text
    assert "key_source" not in text
    assert "native-binary/claude" not in text


def test_claude_failure_format_localizes_timeout() -> None:
    text = format_failure("Claude CLI failed", "Claude turn timed out after 600.0s (hard)")
    assert "超时" in text or "timed out" in text.lower()


def test_claude_narrator_localizes_failed_turn_evidence() -> None:
    narrator = ClaudeFieldNarrator()
    event = ClaudeRuntimeEvent(
        id=1,
        task_key="discord:42:main",
        task_id="abc123",
        thread_id="claude-thread-1",
        turn_id="turn-1",
        platform="discord",
        chat_id="42",
        event_type="turn.failed",
        payload={"error": "Claude turn timed out after 600.0s (hard)"},
        occurred_at=time.time(),
    )
    narration = narrator.narrate(event)
    assert narration is not None
    assert "断流" in narration.text or "超时" in narration.text or "失败" in narration.text


def test_claude_narrator_runtime_labels_do_not_use_app_server() -> None:
    narrator = ClaudeFieldNarrator()
    failed = ClaudeRuntimeEvent(
        id=2,
        task_key="discord:42:main",
        task_id="abc123",
        thread_id="claude-thread-1",
        turn_id="turn-1",
        platform="discord",
        chat_id="42",
        event_type="turn.failed",
        payload={"runtime": "agent_sdk", "error": "Claude SDK idle-timed out after 600s."},
        occurred_at=time.time(),
    )
    timed_out = ClaudeRuntimeEvent(
        id=3,
        task_key="discord:42:main",
        task_id="abc123",
        thread_id="claude-thread-1",
        turn_id="turn-1",
        platform="discord",
        chat_id="42",
        event_type="progress.turn_timed_out",
        payload={"runtime": "agent_sdk", "timeout_seconds": 600},
        occurred_at=time.time(),
    )

    for event in (failed, timed_out):
        narration = narrator.narrate(event)
        assert narration is not None
        rendered = narration.render()
        assert "Claude SDK" in rendered
        assert "app-server" not in rendered


def test_claude_narrator_reports_tool_completed() -> None:
    narrator = ClaudeFieldNarrator()
    event = ClaudeRuntimeEvent(
        id=2,
        task_key="discord:42:main",
        task_id="abc123",
        thread_id="claude-thread-1",
        turn_id="turn-1",
        platform="discord",
        chat_id="42",
        event_type="progress.tool_completed",
        payload={"tool_iterations": 3},
        occurred_at=time.time(),
    )
    narration = narrator.narrate(event)
    assert narration is not None
    assert "3" in narration.render() or "操作" in narration.text


def test_claude_narrator_status_uses_terminal_event_when_present() -> None:
    narrator = ClaudeFieldNarrator()
    now = time.time()
    events = [
        ClaudeRuntimeEvent(
            id=1,
            task_key="discord:42:main",
            task_id="abc123",
            thread_id="claude-thread-1",
            turn_id="turn-1",
            platform="discord",
            chat_id="42",
            event_type="progress.notification",
            payload={"method": "item/started", "item": {"type": "tool_use", "name": "Bash"}},
            occurred_at=now - 30,
        ),
        ClaudeRuntimeEvent(
            id=2,
            task_key="discord:42:main",
            task_id="abc123",
            thread_id="claude-thread-1",
            turn_id="turn-1",
            platform="discord",
            chat_id="42",
            event_type="turn.completed",
            payload={},
            occurred_at=now - 5,
        ),
    ]
    text = narrator.status_text(events, workspace="/repo", thread_id="claude-thread-1")
    assert "收尾" in text or "完成" in text
    assert "最后活动" in text


@pytest.mark.asyncio
async def test_claude_sessions_verbose_shows_thread_id_prefix(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    first = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new task", workspace="/repo")
    )
    verbose = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="sessions verbose", workspace="/repo")
    )
    assert verbose.status == "ok"
    assert first.thread_id[:8] in verbose.text


@pytest.mark.asyncio
async def test_claude_sessions_include_local_claude_jsonl_sessions(tmp_path, monkeypatch) -> None:
    from gateway.control_planes.claude.local_sessions import ClaudeLocalSessionIndex

    repo = _make_git_repo(tmp_path / "repo")
    session_dir = tmp_path / "claude-home" / "projects" / "repo"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "local-session-1.jsonl"
    session_file.write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "local-session-1",
                "timestamp": "2026-06-22T01:02:03+00:00",
                "cwd": str(repo),
                "message": {"content": [{"type": "text", "text": "local task title"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    service = _service(
        tmp_path,
        monkeypatch,
        local_session_index=ClaudeLocalSessionIndex(tmp_path / "claude-home"),
    )

    result = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="sessions",
            workspace=str(repo),
        )
    )

    assert result.status == "ok"
    assert "local task title" in result.text
    assert result.diagnostics["sessions"][0]["thread_id"] == "local-session-1"


@pytest.mark.asyncio
async def test_claude_select_by_index_sets_selected_session(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    first = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new task", workspace="/repo")
    )
    select = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="select 1")
    )
    assert select.status == "ok"
    assert "已选择" in select.text
    selected = service._selected_store.get("discord:42:main")
    assert selected is not None
    assert selected.task_id == first.task_id


@pytest.mark.asyncio
async def test_claude_resume_local_jsonl_session_creates_hermes_record(tmp_path, monkeypatch) -> None:
    from gateway.control_planes.claude.local_sessions import ClaudeLocalSessionIndex

    repo = _make_git_repo(tmp_path / "repo")
    session_dir = tmp_path / "claude-home" / "projects" / "repo"
    session_dir.mkdir(parents=True)
    (session_dir / "local-session-2.jsonl").write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "local-session-2",
                "timestamp": "2026-06-22T01:02:03+00:00",
                "cwd": str(repo),
                "title": "local resumable session",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    service = _service(
        tmp_path,
        monkeypatch,
        local_session_index=ClaudeLocalSessionIndex(tmp_path / "claude-home"),
    )

    result = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="resume 1 follow up",
            workspace=str(repo),
        )
    )

    assert result.status == "completed"
    assert result.thread_id == "local-session-2"
    assert "local:" not in result.task_id
    record = service._registry.get(task_key="discord:42:main")
    assert record is not None
    assert record.thread_id == "local-session-2"


@pytest.mark.asyncio
async def test_claude_resume_by_index_reuses_session(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    first = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new task", workspace="/repo")
    )
    resume = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="resume 1 follow up",
            workspace="/repo",
        )
    )
    assert resume.status == "completed"
    assert resume.thread_id == first.thread_id


@pytest.mark.asyncio
async def test_claude_diff_returns_git_digest_for_workspace(
    tmp_path, monkeypatch
) -> None:
    service = _service(tmp_path, monkeypatch)
    repo = _make_git_repo(tmp_path / "repo")
    monkeypatch.setenv("HERMES_CLAUDE_WORKSPACE_ROOTS", str(tmp_path))
    await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="new task",
            workspace=str(repo),
        )
    )
    diff = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="diff",
            workspace=str(repo),
        )
    )
    assert diff.status in {"ok", "failed"}
    # Repository is a fresh git repo with no commits, so digest is available
    # but empty; we accept either ok (clean) or failed (no commits).
    assert "Git" in diff.text or "git" in diff.text.lower()
