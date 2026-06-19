from types import SimpleNamespace

import pytest

from gateway.control_planes.codex import (
    CodexCommandService,
    CommandRequest,
    build_codex_task_key,
)
from gateway.control_planes.codex.event_store import CodexRuntimeEventStore


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

    assert first.thread_id == second.thread_id
    assert first.task_id == second.task_id
    assert len(service._registry.records) == 1
    assert "Task: 123" in status.text
    assert "Task: 456" not in status.text
    assert "Turn: turn-2" in status.text
    assert sessions.text.count(first.thread_id[:8]) == 1


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
    assert f"Workspace: {repo_a}" in resumed.text
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
    assert "Codex approval bridge unavailable" in result.text


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
    assert options["notification_poll_timeout"] == 0.5
    assert callable(options["progress_callback"])


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
    assert "Codex events" in events_result.text
    assert "codex.notification" in events_result.text


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
    assert "Approve for me" in result.text
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
    assert "Full Access" in result.text
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
    assert "Codex task timed out" in result.text
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
    assert f"Workspace: {repo_b}" in result.text


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
    assert "Default: /default" in telegram_current.text
