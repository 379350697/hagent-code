from types import SimpleNamespace
from datetime import datetime, timezone
import json
import time

import pytest

from gateway.control_planes.codex import (
    CodexCommandService,
    CommandRequest,
    build_codex_task_key,
)
from gateway.control_planes.codex.event_store import CodexRuntimeEvent, CodexRuntimeEventStore
from gateway.control_planes.codex.formatting import format_failure
from gateway.control_planes.codex.narrator import CodexFieldNarrator
from gateway.control_planes.codex.doctor import DoctorCheck


class FakeCodexSession:
    def __init__(
        self,
        *,
        cwd,
        approval_callback=None,
        config_overrides=None,
        resume_thread_id="",
    ):
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.config_overrides = list(config_overrides or [])
        self.resume_thread_id = resume_thread_id
        self.thread_id = resume_thread_id or f"thread-{cwd.rsplit('/', 1)[-1] or 'root'}"
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
            token_usage_total={},
        )

    def request_interrupt(self):
        self.interrupted = True

    def close(self):
        self.closed = True

    def set_approval_callback(self, callback):
        self.approval_callback = callback


class TimeoutCodexSession(FakeCodexSession):
    def ensure_started(self):
        raise TimeoutError("thread/start timed out after 15s")


class ObserverUnconfirmedCodexSession(FakeCodexSession):
    def run_turn(self, user_input, **options):
        self.turns += 1
        self.inputs.append(user_input)
        self.run_turn_options.append(dict(options))
        return SimpleNamespace(
            final_text="",
            error="turn timed out after 600.0s without app-server activity",
            interrupted=True,
            should_retire=True,
            turn_id=f"turn-{self.turns}",
            token_usage_total={},
        )


class ApprovalCodexSession(FakeCodexSession):
    last_instance = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ApprovalCodexSession.last_instance = self

    def run_turn(self, user_input, **options):
        if self.approval_callback is not None:
            try:
                self.approval_callback("touch file.txt", "Codex requests exec in /repo")
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
                    token_usage_total={},
                )
        return super().run_turn(user_input, **options)


class CountingCodexSession(FakeCodexSession):
    instances = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        CountingCodexSession.instances.append(self)


class ProgressCodexSession(FakeCodexSession):
    def run_turn(self, user_input, **options):
        callback = options.get("progress_callback")
        if callback is not None:
            callback({"stage": "turn_started"})
            callback({"stage": "tool_completed", "tool_iterations": 1})
        return super().run_turn(user_input, **options)


class RichProgressCodexSession(FakeCodexSession):
    def run_turn(self, user_input, **options):
        callback = options.get("progress_callback")
        if callback is not None:
            callback({"stage": "turn_started"})
            callback({"stage": "waiting", "idle_seconds": 12})
            callback(
                {
                    "stage": "notification",
                    "method": "thread/tokenUsage/updated",
                    "notification": {
                        "method": "thread/tokenUsage/updated",
                        "params": {"tokenUsage": {"total": {"total": 100}}},
                    },
                }
            )
            callback(
                {
                    "stage": "notification",
                    "method": "item/started",
                    "notification": {
                        "method": "item/started",
                        "params": {
                            "item": {
                                "type": "commandExecution",
                                "command": "rg -n codex gateway",
                                "cwd": "/repo",
                            }
                        },
                    },
                }
            )
            callback(
                {
                    "stage": "notification",
                    "method": "item/completed",
                    "notification": {
                        "method": "item/completed",
                        "params": {
                            "item": {
                                "type": "commandExecution",
                                "command": "pytest tests/gateway",
                                "cwd": "/repo",
                                "aggregatedOutput": "129 passed in 18.55s",
                                "exitCode": 0,
                            }
                        },
                    },
                }
            )
        return super().run_turn(user_input, **options)


class ApprovalProgressCodexSession(FakeCodexSession):
    def run_turn(self, user_input, **options):
        callback = options.get("progress_callback")
        if callback is not None:
            request = {
                "method": "request",
                "params": {"command": "rg -n secret ."},
            }
            callback({"stage": "approval_requested", "request": request})
            callback({"stage": "server_request", "method": "request", "request": request})
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
        from gateway.control_planes.codex.selection import SelectedSession

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
    session_factory=FakeCodexSession,
    *,
    registry=None,
    selected_store=None,
    event_store=None,
):
    from gateway.control_planes.codex import service as service_mod

    monkeypatch.setattr(service_mod, "load_codex_cfg", lambda: {})
    monkeypatch.setattr(service_mod, "read_codex_config_model", lambda: "gpt-5.5")
    return CodexCommandService(
        registry=registry or MemoryRegistry(),
        workspace_store=MemoryWorkspaceStore(),
        selected_store=selected_store or MemorySelectedStore(),
        session_factory=session_factory,
        event_store=event_store or CodexRuntimeEventStore(str(tmp_path / "events.sqlite3")),
    )


def _make_git_repo(path):
    git_dir = path / ".git"
    git_dir.mkdir(parents=True)
    (path / "README.md").write_text("test\n", encoding="utf-8")
    return path


def test_task_key_is_platform_scoped() -> None:
    discord = CommandRequest(platform="discord", chat_id="123", thread_id="", text="")
    telegram = CommandRequest(platform="telegram", chat_id="123", thread_id="", text="")

    assert build_codex_task_key(discord) == "discord:123:main"
    assert build_codex_task_key(telegram) == "telegram:123:main"
    assert build_codex_task_key(discord) != build_codex_task_key(telegram)


@pytest.mark.asyncio
async def test_status_isolated_by_platform_and_chat(tmp_path, monkeypatch) -> None:
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
async def test_continue_requires_live_session(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)

    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="continue more")
    )

    assert result.status == "not_found"
    assert "/codex new" in result.text


@pytest.mark.asyncio
async def test_continue_updates_existing_session_record(tmp_path, monkeypatch) -> None:
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
    verbose_sessions = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="sessions verbose", workspace="/repo")
    )

    assert first.thread_id == second.thread_id
    assert first.task_id == second.task_id
    assert len(service._registry.records) == 1
    assert "任务：123" in status.text
    assert "任务：456" not in status.text
    assert "轮次：turn-2" in status.text
    assert first.task_id not in status.text
    assert first.thread_id not in status.text
    assert first.thread_id[:8] not in sessions.text
    assert verbose_sessions.text.count(first.thread_id[:8]) == 1


@pytest.mark.asyncio
async def test_continue_clears_previous_terminal_fields_while_new_turn_runs(
    tmp_path,
    monkeypatch,
) -> None:
    registry = MemoryRegistry()
    observations = []

    class InspectingContinueSession(FakeCodexSession):
        def run_turn(self, user_input, **options):
            if self.turns >= 1:
                record = registry.get(task_key="discord:42:main")
                observations.append(
                    {
                        "status": record.status,
                        "completed_at": record.completed_at,
                        "token_usage": dict(record.token_usage),
                        "turn_started_at": record.turn_started_at,
                    }
                )
            return super().run_turn(user_input, **options)

    service = _service(
        tmp_path,
        monkeypatch,
        session_factory=InspectingContinueSession,
        registry=registry,
    )

    first = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new 123", workspace="/repo")
    )
    old_completed_at = float(registry.records[first.task_id].completed_at or 0.0)
    registry.records[first.task_id].token_usage = {"totalTokens": 123}

    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="continue 456", workspace="/repo")
    )

    assert observations
    running_state = observations[0]
    assert running_state["status"] == "running"
    assert running_state["completed_at"] is None
    assert running_state["token_usage"] == {}
    assert running_state["turn_started_at"] > old_completed_at


@pytest.mark.asyncio
async def test_continue_restores_selected_session_after_service_restart(
    tmp_path, monkeypatch,
) -> None:
    CountingCodexSession.instances = []
    registry = MemoryRegistry()
    selected = MemorySelectedStore()
    first_service = _service(
        tmp_path,
        monkeypatch,
        session_factory=CountingCodexSession,
        registry=registry,
        selected_store=selected,
    )
    first = await first_service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="new yesterday",
            workspace="/repo-a",
        )
    )

    second_service = _service(
        tmp_path,
        monkeypatch,
        session_factory=CountingCodexSession,
        registry=registry,
        selected_store=selected,
    )
    second = await second_service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="continue today",
            workspace="/repo-b",
        )
    )

    assert second.status == "completed"
    assert second.thread_id == first.thread_id
    assert len(CountingCodexSession.instances) == 2
    assert CountingCodexSession.instances[1].resume_thread_id == first.thread_id
    assert CountingCodexSession.instances[1].cwd == "/repo-a"


@pytest.mark.asyncio
async def test_resume_selector_continues_old_workspace_after_workspace_switch(
    tmp_path, monkeypatch,
) -> None:
    CountingCodexSession.instances = []
    service = _service(tmp_path, monkeypatch, session_factory=CountingCodexSession)
    repo_a = _make_git_repo(tmp_path / "repo-a")
    repo_b = _make_git_repo(tmp_path / "repo-b")

    first = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new first", workspace=str(repo_a))
    )
    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text=f"workspace set {repo_b}")
    )
    second = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new second", workspace=str(repo_a))
    )
    resumed = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text=f"resume {first.task_id} back to first",
            workspace=str(repo_b),
        )
    )

    assert first.thread_id != second.thread_id
    assert resumed.thread_id == first.thread_id
    assert f"工作区：{repo_a}" in resumed.text
    assert CountingCodexSession.instances[-1].resume_thread_id == first.thread_id
    assert CountingCodexSession.instances[-1].cwd == str(repo_a)


@pytest.mark.asyncio
async def test_select_changes_current_session_without_running_turn(tmp_path, monkeypatch) -> None:
    CountingCodexSession.instances = []
    service = _service(tmp_path, monkeypatch, session_factory=CountingCodexSession)
    repo_a = _make_git_repo(tmp_path / "repo-a")
    repo_b = _make_git_repo(tmp_path / "repo-b")

    first = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new first", workspace=str(repo_a))
    )
    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text=f"workspace set {repo_b}")
    )
    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new second", workspace=str(repo_a))
    )
    turns_before = sum(instance.turns for instance in CountingCodexSession.instances)

    selected = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text=f"select {first.task_id}")
    )
    continued = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="continue after select")
    )

    assert selected.status == "ok"
    assert sum(instance.turns for instance in CountingCodexSession.instances) == turns_before + 1
    assert continued.thread_id == first.thread_id


@pytest.mark.asyncio
async def test_selector_ambiguity_returns_candidates_without_running(tmp_path, monkeypatch) -> None:
    from gateway.control_planes.codex.records import make_task_record

    registry = MemoryRegistry()
    registry.upsert(
        make_task_record(
            task_id="abc111",
            task_key="discord:42:main",
            status="completed",
            workspace="/repo-a",
            thread_id="thread-one",
            turn_id="turn-1",
            model="gpt-5.5",
            approval="on-request",
            sandbox="workspace-write",
            plan_mode=False,
            prompt="one",
            last_message="done",
        )
    )
    registry.upsert(
        make_task_record(
            task_id="abc222",
            task_key="discord:42:main",
            status="completed",
            workspace="/repo-b",
            thread_id="thread-two",
            turn_id="turn-2",
            model="gpt-5.5",
            approval="on-request",
            sandbox="workspace-write",
            plan_mode=False,
            prompt="two",
            last_message="done",
        )
    )
    service = _service(tmp_path, monkeypatch, registry=registry)

    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="select abc")
    )

    assert result.status == "ambiguous"
    assert "abc111" in result.text
    assert "abc222" in result.text


@pytest.mark.asyncio
async def test_new_does_not_retire_running_live_session(tmp_path, monkeypatch) -> None:
    CountingCodexSession.instances = []
    service = _service(tmp_path, monkeypatch, session_factory=CountingCodexSession)

    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new first", workspace="/repo-a")
    )
    live = service._sessions.peek("discord:42:main")
    assert live is not None
    assert live.lock.acquire(blocking=False)
    try:
        result = await service.handle(
            CommandRequest(platform="discord", chat_id="42", text="new second", workspace="/repo-b")
        )
    finally:
        live.lock.release()

    assert result.status == "busy"
    assert len(CountingCodexSession.instances) == 1
    assert CountingCodexSession.instances[0].closed is False


@pytest.mark.asyncio
async def test_resume_does_not_retire_running_live_session(tmp_path, monkeypatch) -> None:
    CountingCodexSession.instances = []
    service = _service(tmp_path, monkeypatch, session_factory=CountingCodexSession)
    first = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new first", workspace="/repo-a")
    )
    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new second", workspace="/repo-b")
    )
    live = service._sessions.peek("discord:42:main")
    assert live is not None
    assert live.lock.acquire(blocking=False)
    try:
        result = await service.handle(
            CommandRequest(
                platform="discord",
                chat_id="42",
                text=f"resume {first.task_id} should be busy",
                workspace="/repo-c",
            )
        )
    finally:
        live.lock.release()

    assert result.status == "busy"
    assert CountingCodexSession.instances[-1].closed is False


@pytest.mark.asyncio
async def test_sessions_all_requires_admin_or_diagnostics(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)

    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new first", workspace="/repo")
    )
    forbidden = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="sessions all")
    )
    allowed = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="sessions all", is_admin=True)
    )

    assert forbidden.status == "forbidden"
    assert allowed.status == "ok"


@pytest.mark.asyncio
async def test_sessions_all_allows_explicit_diagnostics_env(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_DIAGNOSTICS", "1")

    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new first", workspace="/repo")
    )
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="sessions all")
    )

    assert result.status == "ok"


@pytest.mark.asyncio
async def test_diff_uses_selected_session_workspace(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    from gateway.control_planes.codex import service as service_mod

    seen = []

    def fake_git_digest(workspace):
        seen.append(workspace)
        return {
            "available": True,
            "repoRoot": workspace,
            "branch": "main",
            "dirty": False,
            "changedFiles": [],
        }

    monkeypatch.setattr(service_mod, "git_digest", fake_git_digest)
    repo_a = _make_git_repo(tmp_path / "repo-a")
    repo_b = _make_git_repo(tmp_path / "repo-b")

    first = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new first", workspace=str(repo_a))
    )
    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text=f"workspace set {repo_b}")
    )
    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new second", workspace=str(repo_a))
    )
    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text=f"select {first.task_id}")
    )
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="diff", workspace=str(repo_b))
    )

    assert result.status == "ok"
    assert seen[-1] == str(repo_a)


@pytest.mark.asyncio
async def test_plan_continues_live_session_when_present(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)

    first = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new implement x", workspace="/repo")
    )
    planned = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="plan add tests", workspace="/repo")
    )

    live = service._sessions.peek("discord:42:main")
    assert live is not None
    assert first.thread_id == planned.thread_id
    assert first.task_id == planned.task_id
    assert len(service._registry.records) == 1
    assert live.session.inputs[0] == "implement x"
    assert "add tests" in live.session.inputs[1]
    assert live.session.inputs[1].startswith("Create a detailed implementation plan first.")


@pytest.mark.asyncio
async def test_plan_session_title_uses_user_task_not_internal_prompt(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)

    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="plan 增加中文进度", workspace="/repo")
    )
    sessions = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="sessions", workspace="/repo")
    )

    assert sessions.status == "ok"
    assert "增加中文进度" in sessions.text
    assert "Create a detailed implementation plan first" not in sessions.text


@pytest.mark.asyncio
async def test_sessions_separate_resumable_session_from_last_turn_status(
    tmp_path, monkeypatch,
) -> None:
    from gateway.control_planes.codex.records import make_task_record

    registry = MemoryRegistry()
    registry.upsert(
        make_task_record(
            task_id="failed123",
            task_key="discord:42:main",
            status="failed",
            workspace="/repo",
            thread_id="thread-failed",
            turn_id="turn-1",
            model="gpt-5.5",
            approval="on-request",
            sandbox="workspace-write",
            plan_mode=False,
            prompt="original",
            title="Create a detailed implementation plan first. Include the files to change",
            last_message="turn timed out",
        )
    )
    service = _service(tmp_path, monkeypatch, registry=registry)

    sessions = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="sessions", workspace="/repo")
    )

    assert sessions.status == "ok"
    assert "可接续" in sessions.text
    assert "最近一轮：失败" in sessions.text
    assert "计划会话" in sessions.text
    assert "failed123" not in sessions.text
    assert "thread-failed" not in sessions.text
    assert "Create a detailed implementation plan first" not in sessions.text


@pytest.mark.asyncio
async def test_sessions_verbose_can_show_internal_ids(tmp_path, monkeypatch) -> None:
    from gateway.control_planes.codex.records import make_task_record

    registry = MemoryRegistry()
    registry.upsert(
        make_task_record(
            task_id="abc123",
            task_key="discord:42:main",
            status="completed",
            workspace="/repo",
            thread_id="thread-visible",
            turn_id="turn-1",
            model="gpt-5.5",
            approval="on-request",
            sandbox="workspace-write",
            plan_mode=False,
            prompt="original",
            last_message="done",
        )
    )
    service = _service(tmp_path, monkeypatch, registry=registry)

    sessions = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="sessions verbose", workspace="/repo")
    )

    assert sessions.status == "ok"
    assert "abc123" in sessions.text
    assert "thread-v" in sessions.text


@pytest.mark.asyncio
async def test_codex_approval_bridge_resolves_exec_request(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch, session_factory=ApprovalCodexSession)
    seen = []

    def notify(approval_data):
        seen.append(approval_data)
        from tools.approval import resolve_gateway_approval

        assert resolve_gateway_approval("approval-session", "once") == 1

    result = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="new write file",
            workspace="/repo",
            approval_session_key="approval-session",
            approval_notify=notify,
        )
    )

    assert result.status == "completed"
    assert seen
    assert seen[0]["command"] == "touch file.txt"


@pytest.mark.asyncio
async def test_codex_approval_bridge_unavailable_is_explicit(
    tmp_path, monkeypatch,
) -> None:
    service = _service(tmp_path, monkeypatch, session_factory=ApprovalCodexSession)

    result = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="new write file",
            workspace="/repo",
            approval_session_key="approval-session",
        )
    )

    assert result.status == "failed"
    assert "Codex 审批通道不可用" in result.text


@pytest.mark.asyncio
async def test_codex_observer_timeout_is_unconfirmed_not_failed(
    tmp_path, monkeypatch,
) -> None:
    service = _service(
        tmp_path,
        monkeypatch,
        session_factory=ObserverUnconfirmedCodexSession,
    )

    result = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="new inspect",
            workspace="/repo",
        )
    )

    assert result.status == "unconfirmed"
    assert "Hermes 未确认本轮结果" in result.text
    assert "Codex 任务失败" not in result.text
    status = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="status")
    )
    assert "最近一轮：未确认" in status.text


@pytest.mark.asyncio
async def test_codex_doctor_reports_health_checks(
    tmp_path, monkeypatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    from gateway.control_planes.codex import service as service_mod

    monkeypatch.setattr(
        service_mod,
        "run_codex_doctor",
        lambda **kwargs: [
            DoctorCheck("Codex CLI", "pass", "/usr/bin/codex"),
            DoctorCheck("当前会话", "warn", "当前聊天还没有选中会话"),
        ],
    )

    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="doctor", workspace="/repo")
    )

    assert result.status == "ok"
    assert "Codex 诊断" in result.text
    assert "通过 · Codex CLI" in result.text
    assert "提醒 · 当前会话" in result.text


@pytest.mark.asyncio
async def test_codex_repair_recovered_preview_and_apply(
    tmp_path, monkeypatch,
) -> None:
    from gateway.control_planes.codex.records import make_task_record

    codex_home = tmp_path / "codex"
    session_dir = codex_home / "sessions" / "2999" / "01" / "01"
    session_dir.mkdir(parents=True)
    native_record = {
        "timestamp": "2999-01-01T00:00:00Z",
        "type": "event_msg",
        "payload": {
            "type": "task_complete",
            "turn_id": "turn-native",
            "last_agent_message": "真实已经完成",
        },
    }
    other_native_record = {
        "timestamp": "2999-01-01T00:00:01Z",
        "type": "event_msg",
        "payload": {
            "type": "task_complete",
            "turn_id": "turn-other",
            "last_agent_message": "其他聊天完成",
        },
    }
    mismatched_native_record = {
        "timestamp": "2999-01-01T00:00:02Z",
        "type": "event_msg",
        "payload": {
            "type": "task_complete",
            "turn_id": "turn-different",
            "last_agent_message": "不该修复旧轮次",
        },
    }
    (session_dir / "rollout-2999-thread-repair.jsonl").write_text(
        json.dumps(native_record) + "\n",
        encoding="utf-8",
    )
    (session_dir / "rollout-2999-thread-other.jsonl").write_text(
        json.dumps(other_native_record) + "\n",
        encoding="utf-8",
    )
    (session_dir / "rollout-2999-thread-mismatch.jsonl").write_text(
        json.dumps(mismatched_native_record) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    registry = MemoryRegistry()
    registry.upsert(
        make_task_record(
            task_id="repair123",
            task_key="discord:42:main",
            status="failed",
            workspace="/repo",
            thread_id="thread-repair",
            turn_id="turn-native",
            model="gpt-5.5",
            approval="on-request",
            sandbox="workspace-write",
            plan_mode=False,
            prompt="original",
            last_message="Hermes observer timeout",
        )
    )
    registry.upsert(
        make_task_record(
            task_id="other123",
            task_key="discord:99:main",
            status="failed",
            workspace="/other",
            thread_id="thread-other",
            turn_id="turn-other",
            model="gpt-5.5",
            approval="on-request",
            sandbox="workspace-write",
            plan_mode=False,
            prompt="other",
            last_message="Hermes observer timeout",
        )
    )
    registry.upsert(
        make_task_record(
            task_id="mismatch123",
            task_key="discord:42:main",
            status="failed",
            workspace="/repo",
            thread_id="thread-mismatch",
            turn_id="turn-old",
            model="gpt-5.5",
            approval="on-request",
            sandbox="workspace-write",
            plan_mode=False,
            prompt="mismatch",
            last_message="Hermes observer timeout",
        )
    )
    service = _service(tmp_path, monkeypatch, registry=registry)

    preview = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="repair recovered")
    )
    applied = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="repair recovered --apply",
            is_admin=True,
        )
    )

    assert preview.status == "preview"
    assert "找到 1 条" in preview.text
    assert "真实已经完成" in preview.text
    assert "repair123" not in preview.text
    assert "thread-repair" not in preview.text
    assert "其他聊天完成" not in preview.text
    assert "不该修复旧轮次" not in preview.text
    assert applied.status == "ok"
    assert registry.records["repair123"].status == "completed"
    assert registry.records["repair123"].turn_id == "turn-native"
    assert registry.records["other123"].status == "failed"
    assert registry.records["mismatch123"].status == "failed"


@pytest.mark.asyncio
async def test_codex_status_reconciles_running_record_from_native_completion(
    tmp_path, monkeypatch,
) -> None:
    from gateway.control_planes.codex.records import make_task_record

    codex_home = tmp_path / "codex"
    session_dir = codex_home / "sessions" / "2999" / "01" / "01"
    session_dir.mkdir(parents=True)
    native_record = {
        "timestamp": "2999-01-01T00:00:00Z",
        "type": "event_msg",
        "payload": {
            "type": "task_complete",
            "turn_id": "turn-native",
            "last_agent_message": "native says done",
        },
    }
    (session_dir / "rollout-2999-thread-running.jsonl").write_text(
        json.dumps(native_record) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    registry = MemoryRegistry()
    registry.upsert(
        make_task_record(
            task_id="running123",
            task_key="discord:42:main",
            status="running",
            workspace="/repo",
            thread_id="thread-running",
            turn_id="turn-native",
            model="gpt-5.5",
            approval="on-request",
            sandbox="workspace-write",
            plan_mode=False,
            prompt="original",
            last_message="Codex: turn started",
        )
    )
    event_store = CodexRuntimeEventStore(str(tmp_path / "events.sqlite3"))
    service = _service(tmp_path, monkeypatch, registry=registry, event_store=event_store)

    status = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="status")
    )

    assert status.status == "completed"
    assert "最近一轮：已完成" in status.text
    assert "native 侧已完成" in status.text
    assert registry.records["running123"].status == "completed"
    assert any(
        event.event_type == "progress.native_reconciled"
        for event in event_store.tail(task_id="running123", limit=10)
    )


@pytest.mark.asyncio
async def test_codex_status_marks_old_running_record_recoverable_stale(
    tmp_path, monkeypatch,
) -> None:
    from gateway.control_planes.codex.records import make_task_record
    from gateway.control_planes.codex import service as service_mod

    registry = MemoryRegistry()
    record = make_task_record(
        task_id="stale123",
        task_key="discord:42:main",
        status="running",
        workspace="/repo",
        thread_id="thread-stale",
        turn_id="turn-stale",
        model="gpt-5.5",
        approval="on-request",
        sandbox="workspace-write",
        plan_mode=False,
        prompt="original",
        last_message="Codex: turn started",
    )
    record.updated_at = time.time() - 60
    registry.upsert(record)
    record.updated_at = time.time() - 60
    monkeypatch.setattr(
        service_mod,
        "load_codex_cfg",
        lambda: {"stale_running_seconds": 1},
    )
    event_store = CodexRuntimeEventStore(str(tmp_path / "events.sqlite3"))
    service = _service(tmp_path, monkeypatch, registry=registry, event_store=event_store)
    monkeypatch.setattr(
        service_mod,
        "load_codex_cfg",
        lambda: {"stale_running_seconds": 1},
    )

    status = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="status")
    )

    assert status.status == "recoverable_stale"
    assert "最近一轮：待恢复" in status.text
    assert "状态不可确认" in status.text
    assert registry.records["stale123"].status == "recoverable_stale"


@pytest.mark.asyncio
async def test_codex_status_recovers_stale_record_using_turn_started_at(
    tmp_path, monkeypatch,
) -> None:
    from gateway.control_planes.codex.records import make_task_record

    codex_home = tmp_path / "codex"
    session_dir = codex_home / "sessions" / "2026" / "06" / "21"
    session_dir.mkdir(parents=True)
    now = time.time()
    native_timestamp = datetime.fromtimestamp(now - 30, timezone.utc).isoformat()
    native_record = {
        "timestamp": native_timestamp,
        "type": "event_msg",
        "payload": {
            "type": "task_complete",
            "turn_id": "turn-stale",
            "last_agent_message": "completed before stale marker",
        },
    }
    (session_dir / "rollout-2026-thread-stale-complete.jsonl").write_text(
        json.dumps(native_record) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    registry = MemoryRegistry()
    record = make_task_record(
        task_id="stalecomplete123",
        task_key="discord:42:main",
        status="recoverable_stale",
        workspace="/repo",
        thread_id="thread-stale-complete",
        turn_id="turn-stale",
        model="gpt-5.5",
        approval="on-request",
        sandbox="workspace-write",
        plan_mode=False,
        prompt="original",
        last_message="Codex 状态不可确认，进入恢复/超时路径",
    )
    record.turn_started_at = now - 60
    record.updated_at = now
    registry.upsert(record)
    record.turn_started_at = now - 60
    record.updated_at = now
    service = _service(tmp_path, monkeypatch, registry=registry)

    status = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="status")
    )

    assert status.status == "completed"
    assert registry.records["stalecomplete123"].status == "completed"
    assert "native 侧已完成" in status.text


@pytest.mark.asyncio
async def test_codex_status_reconciles_native_failed_and_interrupted(
    tmp_path, monkeypatch,
) -> None:
    from gateway.control_planes.codex.records import make_task_record

    codex_home = tmp_path / "codex"
    session_dir = codex_home / "sessions" / "2026" / "06" / "21"
    session_dir.mkdir(parents=True)
    now = time.time()
    failed_record = {
        "timestamp": datetime.fromtimestamp(now - 20, timezone.utc).isoformat(),
        "type": "event_msg",
        "payload": {
            "type": "task_failed",
            "turn_id": "turn-failed",
            "message": "native failure",
        },
    }
    interrupted_record = {
        "timestamp": datetime.fromtimestamp(now - 10, timezone.utc).isoformat(),
        "type": "event_msg",
        "payload": {
            "type": "turn_aborted",
            "turn_id": "turn-interrupted",
            "message": "native aborted",
        },
    }
    (session_dir / "rollout-2026-thread-failed.jsonl").write_text(
        json.dumps(failed_record) + "\n",
        encoding="utf-8",
    )
    (session_dir / "rollout-2026-thread-interrupted.jsonl").write_text(
        json.dumps(interrupted_record) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    registry = MemoryRegistry()
    for task_id, thread_id, turn_id in (
        ("failed123", "thread-failed", "turn-failed"),
        ("interrupted123", "thread-interrupted", "turn-interrupted"),
    ):
        record = make_task_record(
            task_id=task_id,
            task_key=f"discord:42:{task_id}",
            status="running",
            workspace="/repo",
            thread_id=thread_id,
            turn_id=turn_id,
            model="gpt-5.5",
            approval="on-request",
            sandbox="workspace-write",
            plan_mode=False,
            prompt="original",
            last_message="Codex: turn started",
        )
        record.turn_started_at = now - 60
        registry.upsert(record)
        registry.records[task_id].turn_started_at = now - 60
    service = _service(tmp_path, monkeypatch, registry=registry)

    failed = await service.handle(
        CommandRequest(platform="discord", chat_id="42", thread_id="failed123", text="status")
    )
    interrupted = await service.handle(
        CommandRequest(platform="discord", chat_id="42", thread_id="interrupted123", text="status")
    )

    assert failed.status == "failed"
    assert "native 侧已确认失败" in failed.text
    assert interrupted.status == "interrupted"
    assert "native 侧已确认中断" in interrupted.text


def test_codex_watchdog_sweep_reconciles_without_status_command(
    tmp_path, monkeypatch,
) -> None:
    from gateway.control_planes.codex.records import make_task_record

    codex_home = tmp_path / "codex"
    session_dir = codex_home / "sessions" / "2026" / "06" / "21"
    session_dir.mkdir(parents=True)
    now = time.time()
    native_record = {
        "timestamp": datetime.fromtimestamp(now - 10, timezone.utc).isoformat(),
        "type": "event_msg",
        "payload": {
            "type": "task_complete",
            "turn_id": "turn-watch",
            "last_agent_message": "watchdog recovered",
        },
    }
    (session_dir / "rollout-2026-thread-watch.jsonl").write_text(
        json.dumps(native_record) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    registry = MemoryRegistry()
    record = make_task_record(
        task_id="watch123",
        task_key="discord:42:main",
        status="running",
        workspace="/repo",
        thread_id="thread-watch",
        turn_id="turn-watch",
        model="gpt-5.5",
        approval="on-request",
        sandbox="workspace-write",
        plan_mode=False,
        prompt="original",
        last_message="Codex: turn started",
    )
    record.turn_started_at = now - 60
    registry.upsert(record)
    registry.records["watch123"].turn_started_at = now - 60
    service = _service(tmp_path, monkeypatch, registry=registry)

    changed = service.sweep_stale_tasks()

    assert changed == 1
    assert registry.records["watch123"].status == "completed"


@pytest.mark.asyncio
async def test_codex_sandbox_config_is_passed_to_app_server(
    tmp_path, monkeypatch,
) -> None:
    service = _service(tmp_path, monkeypatch, session_factory=ApprovalCodexSession)
    from gateway.control_planes.codex import service as service_mod

    monkeypatch.setattr(
        service_mod,
        "load_codex_cfg",
        lambda: {"sandbox": "readonly", "approval_policy": "on-request"},
    )

    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new inspect", workspace="/repo")
    )

    assert ApprovalCodexSession.last_instance is not None
    assert ApprovalCodexSession.last_instance.config_overrides == [
        "-c",
        'sandbox_mode="read-only"',
        "-c",
        'approval_policy="on-request"',
    ]


@pytest.mark.asyncio
async def test_codex_turn_timeouts_are_passed_to_app_server(
    tmp_path, monkeypatch,
) -> None:
    CountingCodexSession.instances = []
    service = _service(tmp_path, monkeypatch, session_factory=CountingCodexSession)
    from gateway.control_planes.codex import service as service_mod

    monkeypatch.setattr(
        service_mod,
        "load_codex_cfg",
        lambda: {
            "turn_timeout_seconds": 2400,
            "post_tool_quiet_timeout_seconds": 45,
            "active_tool_timeout_seconds": 7200,
            "notification_poll_timeout_seconds": 0.5,
        },
    )

    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new inspect", workspace="/repo")
    )

    assert result.status == "completed"
    options = CountingCodexSession.instances[0].run_turn_options[0]
    assert options["turn_timeout"] == 2400.0
    assert options["post_tool_quiet_timeout"] == 45.0
    assert options["active_tool_timeout"] == 7200.0
    assert options["notification_poll_timeout"] == 0.5
    assert options["unbounded_command_policy"] == "conditional_hard"
    assert callable(options["progress_callback"])


@pytest.mark.asyncio
async def test_codex_unbounded_command_policy_is_passed_to_app_server(
    tmp_path, monkeypatch,
) -> None:
    CountingCodexSession.instances = []
    service = _service(tmp_path, monkeypatch, session_factory=CountingCodexSession)
    from gateway.control_planes.codex import service as service_mod

    monkeypatch.setattr(
        service_mod,
        "load_codex_cfg",
        lambda: {"unbounded_command_policy": "strict_hard"},
    )

    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new inspect", workspace="/repo")
    )

    assert result.status == "completed"
    options = CountingCodexSession.instances[0].run_turn_options[0]
    assert options["unbounded_command_policy"] == "strict_hard"


@pytest.mark.asyncio
async def test_codex_progress_notify_reports_turn_activity(
    tmp_path, monkeypatch,
) -> None:
    service = _service(tmp_path, monkeypatch, session_factory=ProgressCodexSession)
    progress = []

    result = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="new inspect",
            workspace="/repo",
            progress_notify=lambda data: progress.append(data),
        )
    )

    assert result.status == "completed"
    assert progress
    assert progress[0]["type"] == "codex_progress"
    assert progress[0]["stage"] == "turn_started"
    assert "我开始接这轮任务了" in progress[0]["text"]
    assert "证据：" in progress[0]["text"]


@pytest.mark.asyncio
async def test_codex_runtime_events_are_persisted_and_queryable(
    tmp_path, monkeypatch,
) -> None:
    event_store = CodexRuntimeEventStore(str(tmp_path / "events.sqlite3"))
    service = _service(
        tmp_path,
        monkeypatch,
        session_factory=RichProgressCodexSession,
        event_store=event_store,
    )
    progress = []

    result = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="new inspect",
            workspace="/repo",
            progress_notify=lambda data: progress.append(data),
        )
    )

    assert result.status == "completed"
    events = event_store.tail(task_key="discord:42:main", task_id=result.task_id, limit=20)
    assert [event.event_type for event in events]
    assert any(event.event_type == "codex.notification" for event in events)
    assert any(event.event_type == "usage.updated" for event in events)
    assert any("刚跑完一轮测试" in item["text"] for item in progress)
    assert not any(item.get("event_type") == "usage.updated" for item in progress)

    events_result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="events", workspace="/repo")
    )
    assert events_result.status == "ok"
    assert "Codex 事件" in events_result.text
    assert "codex.notification" in events_result.text


def test_codex_failure_format_localizes_post_tool_silence() -> None:
    text = format_failure(
        "Codex app-server failed",
        "codex went silent for 90s after a tool result; retiring app-server session.",
    )

    assert "Codex app-server 在上一次操作完成后 90 秒没有新事件" in text
    assert "went silent" not in text


def test_codex_narrator_localizes_failed_turn_evidence() -> None:
    event = CodexRuntimeEvent(
        id=1,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="turn.failed",
        payload={
            "error": "codex went silent for 90s after a tool result; retiring app-server session.",
        },
        occurred_at=0.0,
    )

    narration = CodexFieldNarrator().narrate(event, workspace="/repo", thread_id="thread")

    assert narration is not None
    rendered = narration.render()
    assert "app-server 断流" in rendered
    assert "没有新事件" in rendered
    assert "went silent" not in rendered
    assert "收口" not in rendered


def test_codex_narrator_keeps_observer_timeout_unconfirmed() -> None:
    event = CodexRuntimeEvent(
        id=1,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="turn.unconfirmed",
        payload={
            "error": "turn timed out after 600.0s without app-server activity",
        },
        occurred_at=0.0,
    )

    narration = CodexFieldNarrator().narrate(event, workspace="/repo", thread_id="thread")

    assert narration is not None
    rendered = narration.render()
    assert "没确认到" in rendered
    assert "判失败" in rendered


def test_codex_narrator_marks_interactive_python_command() -> None:
    event = CodexRuntimeEvent(
        id=1,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="codex.notification",
        payload={
            "stage": "notification",
            "method": "item/started",
            "notification": {
                "method": "item/started",
                "params": {
                    "item": {
                        "type": "commandExecution",
                        "id": "ex1",
                        "command": "/bin/bash -lc python3",
                        "cwd": "/home/wl/projects/Lightld",
                    }
                },
            },
        },
        occurred_at=0.0,
    )

    narration = CodexFieldNarrator().narrate(event, workspace="/repo", thread_id="thread")

    assert narration is not None
    rendered = narration.render()
    assert "交互式命令会话" in rendered
    assert "交互式 Python 会话" in rendered
    assert "跑命令验证现场" not in rendered


def test_codex_narrator_reports_command_completion_without_tool_step_wording() -> None:
    event = CodexRuntimeEvent(
        id=1,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="codex.notification",
        payload={
            "stage": "notification",
            "method": "item/completed",
            "notification": {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "commandExecution",
                        "command": "rg -n codex gateway",
                        "cwd": "/repo",
                    }
                },
            },
        },
        occurred_at=0.0,
    )

    narration = CodexFieldNarrator().narrate(event, workspace="/repo", thread_id="thread")

    assert narration is not None
    rendered = narration.render()
    assert "命令验证" in rendered
    assert "rg -n codex gateway" in rendered
    assert "工具步骤" not in rendered


def test_codex_narrator_rolls_recent_activity_into_generic_progress() -> None:
    command_started = CodexRuntimeEvent(
        id=1,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="codex.notification",
        payload={
            "stage": "notification",
            "method": "item/started",
            "notification": {
                "method": "item/started",
                "params": {
                    "item": {
                        "type": "commandExecution",
                        "command": "rg -n codex gateway",
                        "cwd": "/repo",
                    }
                },
            },
        },
        occurred_at=1.0,
    )
    test_completed = CodexRuntimeEvent(
        id=2,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="codex.notification",
        payload={
            "stage": "notification",
            "method": "item/completed",
            "notification": {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "commandExecution",
                        "command": "pytest control_plane/tests/gateway",
                        "cwd": "/repo",
                        "aggregatedOutput": "151 passed in 21.13s",
                    }
                },
            },
        },
        occurred_at=2.0,
    )
    tool_completed = CodexRuntimeEvent(
        id=3,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="progress.tool_completed",
        payload={"stage": "tool_completed", "tool_iterations": 35},
        occurred_at=3.0,
    )

    narration = CodexFieldNarrator().narrate(
        tool_completed,
        recent_events=[command_started, test_completed, tool_completed],
        workspace="/repo",
        thread_id="thread",
    )

    assert narration is not None
    rendered = narration.render()
    assert "刚完成一次操作" in rendered
    assert "测试完成：151 passed" in rendered
    assert "正在执行命令：rg -n codex gateway" in rendered
    assert "已完成 35 个工具步骤" not in rendered
    assert "工具步骤" not in rendered


def test_codex_narrator_status_uses_persisted_file_change_context() -> None:
    file_change = CodexRuntimeEvent(
        id=1,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="codex.notification",
        payload={
            "stage": "notification",
            "method": "item/completed",
            "notification": {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "fileChange",
                        "changes": [
                            {"path": "control_plane/gateway/control_planes/codex/narrator.py"},
                            {"path": "control_plane/tests/gateway/test_codex_command_service.py"},
                        ],
                    }
                },
            },
        },
        occurred_at=time.time() - 3,
    )
    tool_completed = CodexRuntimeEvent(
        id=2,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="progress.tool_completed",
        payload={"stage": "tool_completed", "tool_iterations": 1},
        occurred_at=time.time() - 1,
    )

    text = CodexFieldNarrator().status_text(
        [file_change, tool_completed],
        workspace="/repo",
        thread_id="thread",
    )

    assert "刚完成一次操作" in text
    assert "文件修改：" in text
    assert "narrator.py" in text
    assert "工具步骤" not in text


def test_codex_narrator_caps_rolling_activity_evidence() -> None:
    long_path = "control_plane/" + "very_long_directory_name/" * 10 + "narrator.py"
    file_change = CodexRuntimeEvent(
        id=1,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="codex.notification",
        payload={
            "stage": "notification",
            "method": "item/completed",
            "notification": {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "fileChange",
                        "changes": [{"path": long_path}],
                    }
                },
            },
        },
        occurred_at=time.time() - 3,
    )
    tool_completed = CodexRuntimeEvent(
        id=2,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="progress.tool_completed",
        payload={"stage": "tool_completed", "tool_iterations": 1},
        occurred_at=time.time() - 1,
    )

    text = CodexFieldNarrator().status_text(
        [file_change, tool_completed],
        workspace="/repo",
        thread_id="thread",
    )

    evidence_lines = [line for line in text.splitlines() if line.startswith("证据：")]
    assert evidence_lines
    assert len(evidence_lines[0]) <= 170
    assert evidence_lines[0].endswith("...")


def test_codex_narrator_reports_active_tool_waiting() -> None:
    event = CodexRuntimeEvent(
        id=1,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="progress.waiting",
        payload={
            "stage": "waiting",
            "idle_seconds": 91,
            "active_tool_label": "/bin/bash -lc python3",
            "active_tool_elapsed_seconds": 180,
        },
        occurred_at=0.0,
    )

    narration = CodexFieldNarrator().narrate(event, workspace="/repo", thread_id="thread")

    assert narration is not None
    rendered = narration.render()
    assert "当前命令还没返回" in rendered
    assert "无新 app-server 事件 1 分钟" in rendered
    assert "/bin/bash -lc python3" in rendered


def test_codex_narrator_reports_unbounded_command_guardrail() -> None:
    event = CodexRuntimeEvent(
        id=1,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="progress.unbounded_command_detected",
        payload={
            "stage": "unbounded_command_detected",
            "command": "journalctl -f -u lightld.service",
            "reason": "持续跟随日志",
            "recommendation": "使用 -n 120 --no-pager",
            "blocked": True,
        },
        occurred_at=0.0,
    )

    narration = CodexFieldNarrator().narrate(event, workspace="/repo", thread_id="thread")

    assert narration is not None
    rendered = narration.render()
    assert "护栏拦截" in rendered
    assert "journalctl -f" in rendered


def test_codex_narrator_status_prioritizes_native_recovery_over_old_waiting() -> None:
    waiting = CodexRuntimeEvent(
        id=1,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="progress.waiting",
        payload={
            "stage": "waiting",
            "idle_seconds": 90,
            "active_tool_label": "journalctl -f",
            "active_tool_elapsed_seconds": 120,
        },
        occurred_at=10.0,
    )
    recovered = CodexRuntimeEvent(
        id=2,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="progress.native_reconciled",
        payload={
            "stage": "native_reconciled",
            "native_reconcile_status": "completed",
            "native_reconcile_reason": "native task_complete confirmed",
        },
        occurred_at=5.0,
    )

    text = CodexFieldNarrator().status_text(
        [recovered, waiting],
        workspace="/repo",
        thread_id="thread",
    )

    assert "native 侧已完成" in text
    assert "当前命令还没返回" not in text


def test_codex_narrator_tolerates_dirty_waiting_durations() -> None:
    event = CodexRuntimeEvent(
        id=1,
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        turn_id="turn",
        platform="discord",
        chat_id="42",
        event_type="progress.waiting",
        payload={
            "stage": "waiting",
            "idle_seconds": "not-a-number",
            "active_tool_label": "python3",
            "active_tool_elapsed_seconds": None,
        },
        occurred_at=0.0,
    )

    narration = CodexFieldNarrator().narrate(event, workspace="/repo", thread_id="thread")

    assert narration is not None
    rendered = narration.render()
    assert "运行 0 秒" in rendered
    assert "无新 app-server 事件 0 秒" in rendered


@pytest.mark.asyncio
async def test_codex_runtime_event_payload_is_redacted(
    tmp_path, monkeypatch,
) -> None:
    event_store = CodexRuntimeEventStore(str(tmp_path / "events.sqlite3"))

    stored = event_store.append(
        task_key="discord:42:main",
        task_id="task",
        thread_id="thread",
        event_type="codex.notification",
        payload={
            "token": "secret-token",
            "message": "Authorization: Bearer abc.def and OPENAI_API_KEY=sk-testsecret",
            "long": "x" * 12_000,
        },
    )

    assert stored.payload["token"] == "[REDACTED]"
    assert "Authorization:[REDACTED]" in stored.payload["message"]
    assert "abc.def" not in stored.payload["message"]
    assert "OPENAI_API_KEY=[REDACTED]" in stored.payload["message"]
    assert str(stored.payload["long"]).endswith("...<truncated>")


@pytest.mark.asyncio
async def test_codex_approval_progress_is_deduped(
    tmp_path, monkeypatch,
) -> None:
    service = _service(tmp_path, monkeypatch, session_factory=ApprovalProgressCodexSession)
    progress = []

    result = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="new inspect",
            workspace="/repo",
            progress_notify=lambda data: progress.append(data),
        )
    )

    assert result.status == "completed"
    approval_messages = [item for item in progress if "现在卡在审批" in item["text"]]
    assert len(approval_messages) == 1
    assert "待执行 rg -n secret ." in approval_messages[0]["text"]


@pytest.mark.asyncio
async def test_codex_danger_mode_auto_approves_without_notify(
    tmp_path, monkeypatch,
) -> None:
    service = _service(tmp_path, monkeypatch, session_factory=ApprovalCodexSession)
    from gateway.control_planes.codex import service as service_mod

    monkeypatch.setattr(
        service_mod,
        "load_codex_cfg",
        lambda: {"sandbox": "danger-full-access", "approval_policy": "never"},
    )

    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new write", workspace="/repo")
    )

    assert result.status == "completed"
    assert ApprovalCodexSession.last_instance.config_overrides == [
        "-c",
        'sandbox_mode="danger-full-access"',
        "-c",
        'approval_policy="never"',
    ]


@pytest.mark.asyncio
async def test_codex_approve_for_me_auto_reviews_without_notify(
    tmp_path, monkeypatch,
) -> None:
    service = _service(tmp_path, monkeypatch, session_factory=ApprovalCodexSession)
    from gateway.control_planes.codex import service as service_mod

    notified = []
    monkeypatch.setattr(
        service_mod,
        "load_codex_cfg",
        lambda: {
            "sandbox": "workspace-write",
            "approval_policy": "on-request",
            "approvals_reviewer": "auto_review",
        },
    )

    result = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="new write",
            workspace="/repo",
            approval_session_key="approval-session",
            approval_notify=lambda data: notified.append(data),
        )
    )

    assert result.status == "completed"
    assert notified == []
    assert ApprovalCodexSession.last_instance.config_overrides == [
        "-c",
        'sandbox_mode="workspace-write"',
        "-c",
        'approval_policy="on-request"',
        "-c",
        'approvals_reviewer="auto_review"',
    ]


@pytest.mark.asyncio
async def test_codex_permissions_match_desktop_profiles(
    tmp_path, monkeypatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    saved = {}

    def fake_save_config_value(key, value):
        saved[key] = value

    import cli

    monkeypatch.setattr(cli, "save_config_value", fake_save_config_value)

    result = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="permissions approve-for-me",
            workspace="/repo",
        )
    )

    assert result.status == "ok"
    assert "自动审批" in result.text
    assert "工作区可写 / 按需审批" in result.text
    assert saved == {
        "codex_app_server.sandbox": "workspace-write",
        "codex_app_server.approval_policy": "on-request",
        "codex_app_server.approvals_reviewer": "auto_review",
    }


@pytest.mark.asyncio
async def test_codex_permissions_full_access_profile(
    tmp_path, monkeypatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    saved = {}

    def fake_save_config_value(key, value):
        saved[key] = value

    import cli

    monkeypatch.setattr(cli, "save_config_value", fake_save_config_value)

    result = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text="permissions full-access",
            workspace="/repo",
        )
    )

    assert result.status == "ok"
    assert "完全访问 / 无需审批" in result.text
    assert saved == {
        "codex_app_server.sandbox": "danger-full-access",
        "codex_app_server.approval_policy": "never",
        "codex_app_server.approvals_reviewer": "",
    }


@pytest.mark.asyncio
async def test_workspace_change_drives_next_new_session(tmp_path, monkeypatch) -> None:
    CountingCodexSession.instances = []
    service = _service(tmp_path, monkeypatch, session_factory=CountingCodexSession)
    repo_a = _make_git_repo(tmp_path / "repo-a")
    repo_b = _make_git_repo(tmp_path / "repo-b")

    first = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new first", workspace=str(repo_a))
    )
    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text=f"workspace set {repo_b}")
    )
    second = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new second", workspace=str(repo_a))
    )

    assert first.status == "completed"
    assert second.status == "completed"
    assert len(CountingCodexSession.instances) == 2
    assert CountingCodexSession.instances[0].cwd == str(repo_a)
    assert CountingCodexSession.instances[1].cwd == str(repo_b)


@pytest.mark.asyncio
async def test_sandbox_config_change_recreates_live_session(tmp_path, monkeypatch) -> None:
    CountingCodexSession.instances = []
    service = _service(tmp_path, monkeypatch, session_factory=CountingCodexSession)
    from gateway.control_planes.codex import service as service_mod

    cfg = {"sandbox": "workspace-write", "approval_policy": "on-request"}
    monkeypatch.setattr(service_mod, "load_codex_cfg", lambda: dict(cfg))

    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new first", workspace="/repo")
    )
    cfg["sandbox"] = "readonly"
    await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="plan second", workspace="/repo")
    )

    assert len(CountingCodexSession.instances) == 2
    assert CountingCodexSession.instances[0].config_overrides == [
        "-c",
        'sandbox_mode="workspace-write"',
        "-c",
        'approval_policy="on-request"',
    ]
    assert CountingCodexSession.instances[1].config_overrides == [
        "-c",
        'sandbox_mode="read-only"',
        "-c",
        'approval_policy="on-request"',
    ]


@pytest.mark.asyncio
async def test_codex_start_timeout_is_reported_without_platform_failure(
    tmp_path, monkeypatch,
) -> None:
    service = _service(tmp_path, monkeypatch, session_factory=TimeoutCodexSession)

    result = await service.handle(
        CommandRequest(platform="telegram", chat_id="42", text="new test", workspace="/repo")
    )

    assert result.status == "failed"
    assert "Codex 任务超时" in result.text
    assert result.diagnostics["phase"] == "thread/start"


@pytest.mark.asyncio
async def test_workspace_list_and_selection_drive_new_session(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    repo_a = _make_git_repo(tmp_path / "repo-a")
    repo_b = _make_git_repo(tmp_path / "repo-b")
    monkeypatch.setenv("HERMES_CODEX_WORKSPACE_ROOTS", str(tmp_path))

    listing = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="workspace")
    )

    assert listing.status == "ok"
    assert str(repo_a) in listing.text
    assert str(repo_b) in listing.text

    selected = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="workspace set 2")
    )
    result = await service.handle(
        CommandRequest(platform="discord", chat_id="42", text="new inspect workspace")
    )

    assert selected.status == "ok"
    assert str(repo_b) in selected.text
    assert f"工作区：{repo_b}" in result.text


@pytest.mark.asyncio
async def test_workspace_selection_is_platform_scoped(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    repo = _make_git_repo(tmp_path / "same-chat-repo")

    discord_selected = await service.handle(
        CommandRequest(
            platform="discord",
            chat_id="42",
            text=f"workspace set {repo}",
            workspace="/default",
        )
    )
    telegram_current = await service.handle(
        CommandRequest(
            platform="telegram",
            chat_id="42",
            text="workspace current",
            workspace="/default",
        )
    )

    assert discord_selected.status == "ok"
    assert str(repo) in discord_selected.text
    assert "默认：/default" in telegram_current.text
