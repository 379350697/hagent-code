from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from agent.transports.claude_agent_sdk_session import (
    ClaudeAgentSdkSession,
    _parse_sdk_message,
    _redact_raw_line,
)


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    name: str
    input: dict[str, Any]


@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any]


@dataclass
class AssistantMessage:
    content: list[Any]
    model: str = "deepseek-v4-pro"
    usage: dict[str, Any] | None = None


@dataclass
class ResultMessage:
    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    result: str | None = None
    usage: dict[str, Any] | None = None
    api_error_status: int | None = None
    errors: list[str] | None = None


class FakeRunner:
    def __init__(self, messages):
        self.messages = list(messages)
        self.calls = []
        self.interrupted = False

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["interrupt_state"]["client"] = self
        kwargs["interrupt_state"]["loop"] = __import__("asyncio").get_running_loop()
        for message in self.messages:
            yield message

    async def interrupt(self):
        self.interrupted = True


class SlowFirstMessageRunner(FakeRunner):
    async def run(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0.7)
        for message in self.messages:
            yield message


def _profile():
    return {
        "name": "opencodego",
        "base_url": "http://127.0.0.1:15721",
        "api_key": "sk-test-secret",
        "api_key_source": "test",
        "model": "deepseek-v4-pro",
        "effort": "xhigh",
    }


def test_sdk_parser_extracts_system_session() -> None:
    parsed = _parse_sdk_message(
        SystemMessage(subtype="init", data={"session_id": "sdk-session-1"}),
        "",
    )
    assert parsed.session_id == "sdk-session-1"
    assert parsed.progress_events[0]["subtype"] == "init"


def test_sdk_parser_extracts_assistant_text_tool_and_usage() -> None:
    parsed = _parse_sdk_message(
        AssistantMessage(
            content=[TextBlock("hello"), ToolUseBlock("Bash", {"command": "pwd"})],
            usage={"input_tokens": 10, "output_tokens": 5},
        ),
        "",
    )
    assert parsed.delta_text == "hello"
    assert parsed.tool_count == 1
    assert parsed.usage["total_tokens"] == 15


def test_sdk_session_runs_successful_turn() -> None:
    runner = FakeRunner([
        SystemMessage(subtype="init", data={"session_id": "sdk-session-1"}),
        AssistantMessage(content=[TextBlock("done")]),
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sdk-session-1",
            result="done",
            usage={"input_tokens": 20, "output_tokens": 6},
        ),
    ])
    session = ClaudeAgentSdkSession(
        cwd="/tmp",
        sdk_profile_config=_profile(),
        runner=runner,
        permission_mode="plan",
    )

    result = session.run_turn("hello")

    assert result.error is None
    assert result.final_text == "done"
    assert result.session_id == "sdk-session-1"
    assert result.runtime == "agent_sdk"
    assert result.runtime_profile == "opencodego"
    assert runner.calls[0]["session_id"] == ""


def test_sdk_session_resumes_existing_session() -> None:
    runner = FakeRunner([
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sdk-session-1",
            result="ok",
        ),
    ])
    session = ClaudeAgentSdkSession(
        cwd="/tmp",
        resume_thread_id="sdk-session-1",
        sdk_profile_config=_profile(),
        runner=runner,
    )

    result = session.run_turn("continue")

    assert result.session_id == "sdk-session-1"
    assert runner.calls[0]["session_id"] == "sdk-session-1"


def test_sdk_session_poll_timeout_does_not_cancel_stream() -> None:
    runner = SlowFirstMessageRunner([
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sdk-session-1",
            result="ok",
        ),
    ])
    session = ClaudeAgentSdkSession(cwd="/tmp", sdk_profile_config=_profile(), runner=runner)

    result = session.run_turn("hello", turn_timeout=3, idle_timeout=3)

    assert result.error is None
    assert result.final_text == "ok"


def test_sdk_session_classifies_result_error() -> None:
    runner = FakeRunner([
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="sdk-session-1",
            result="rate limited",
            api_error_status=429,
        ),
    ])
    session = ClaudeAgentSdkSession(cwd="/tmp", sdk_profile_config=_profile(), runner=runner)

    result = session.run_turn("hello")

    assert result.error == "rate limited"
    assert result.error_kind == "sdk_rate_limit"


def test_sdk_session_preflight_missing_key() -> None:
    session = ClaudeAgentSdkSession(
        cwd="/tmp",
        sdk_profile_config={"name": "opencodego", "api_key_env": "NO_SUCH_KEY"},
        runner=FakeRunner([]),
    )
    result = session.run_turn("hello")
    assert result.error_kind == "sdk_preflight_failed"
    assert not result.started


def test_sdk_raw_tail_redacts_secrets() -> None:
    redacted = _redact_raw_line("Authorization: Bearer abc token=sk-secret123")
    assert "abc" not in redacted
    assert "sk-secret123" not in redacted
    assert "[REDACTED]" in redacted
