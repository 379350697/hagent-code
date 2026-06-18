from types import SimpleNamespace

import pytest

from gateway.control_planes.codex import (
    CodexCommandService,
    CommandRequest,
    build_codex_task_key,
)


class FakeCodexSession:
    def __init__(self, *, cwd, approval_callback=None):
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.thread_id = f"thread-{cwd.rsplit('/', 1)[-1] or 'root'}"
        self.interrupted = False
        self.closed = False
        self.turns = 0

    def ensure_started(self):
        return self.thread_id

    def run_turn(self, user_input):
        self.turns += 1
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


class TimeoutCodexSession(FakeCodexSession):
    def ensure_started(self):
        raise TimeoutError("thread/start timed out after 15s")


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


def _service(tmp_path, monkeypatch, session_factory=FakeCodexSession):
    from gateway.control_planes.codex import service as service_mod

    monkeypatch.setattr(service_mod, "load_codex_cfg", lambda: {})
    monkeypatch.setattr(service_mod, "read_codex_config_model", lambda: "gpt-5.5")
    return CodexCommandService(
        registry=MemoryRegistry(),
        session_factory=session_factory,
    )


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
