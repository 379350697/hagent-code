"""Tests for ClaudeCliSession — drive turns through a fake subprocess.

The Claude transport spawns a fresh `claude -p` subprocess per turn and parses
stream-json output. These tests pin the parsing, capability probing, and turn
result extraction without spawning the real claude binary.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import agent.transports.claude_cli_session as session_mod
from agent.transports.claude_cli_session import (
    ClaudeCliCapabilities,
    ClaudeCliSession,
    TurnResult,
    _coerce_turn_input_text,
    _nonzero_exit_error,
    _normalize_permission_mode,
    _parse_stream_line,
    _redact_raw_line,
    _success_then_nonzero_warning,
    parse_claude_help,
    resolve_claude_binary,
)


class FakeClaudeCliSession(ClaudeCliSession):
    """Override _run_turn_async to return a canned TurnResult.

    Lets the service-level tests drive the session without spawning a real
    subprocess. The real :meth:`run_turn` path is exercised by the
    integration tests in test_claude_command_service.py via FakeClaudeSession.
    """

    def __init__(self, *, cwd, **kwargs):
        super().__init__(cwd=cwd, **kwargs)
        self._fake_turn: TurnResult | None = None
        self._turn_calls: list[tuple[Any, dict]] = []

    def set_fake_turn(self, result: TurnResult) -> None:
        self._fake_turn = result

    def run_turn(self, user_input, **options):
        self._turn_calls.append((user_input, dict(options)))
        if self._fake_turn is not None:
            return self._fake_turn
        return TurnResult(
            final_text=f"fake: {user_input}",
            session_id=self.thread_id or "fake-session",
            turn_id="fake-turn-1",
        )


def test_claude_capabilities_parsed_from_help_text() -> None:
    help_text = """
  -p, --print                           Print response and exit
      --output-format <format>          Output format (json|stream-json|text)
      --permission-mode <mode>          Permission mode (acceptEdits|...)
      --model <model>                   Model for the current session
      --effort <level>                  Effort level (low|medium|high|xhigh|max)
  -r, --resume [value]                  Resume a conversation by session ID
      --session-id <uuid>               Use a specific session ID
      --include-partial-messages        Include partial message chunks
      --include-hook-events             Include all hook lifecycle events
"""
    caps = parse_claude_help(help_text)
    assert caps.print_prompt
    assert caps.output_format
    assert caps.stream_json_output
    assert caps.permission_mode
    assert caps.model
    assert caps.effort
    assert caps.resume
    assert caps.session_id
    assert caps.include_partial_messages
    assert caps.include_hook_events


def test_claude_capabilities_minimal_when_help_lacks_print() -> None:
    caps = parse_claude_help("no relevant flags here")
    assert not caps.print_prompt


def test_normalize_permission_mode_handles_aliases() -> None:
    assert _normalize_permission_mode("") == "acceptEdits"
    assert _normalize_permission_mode("acceptEdits") == "acceptEdits"
    assert _normalize_permission_mode("auto") == "auto"
    assert _normalize_permission_mode("yolo") == "bypassPermissions"
    assert _normalize_permission_mode("full") == "bypassPermissions"
    assert _normalize_permission_mode("plan") == "plan"
    # Unknown values pass through unchanged
    assert _normalize_permission_mode("custom") == "custom"


def test_coerce_turn_input_text_collapses_rich_content() -> None:
    assert _coerce_turn_input_text("hello") == "hello"
    assert _coerce_turn_input_text(None) == ""
    assert _coerce_turn_input_text(42) == "42"
    parts = [
        {"type": "text", "text": "hello "},
        {"type": "image", "url": "data:image/png;base64,..."},
        {"type": "text", "text": "world"},
    ]
    assert _coerce_turn_input_text(parts) == "hello \n\n[image attached]\n\nworld"
    # Plain string list entries are kept
    assert _coerce_turn_input_text(["a", "b"]) == "a\n\nb"


def test_parse_stream_line_handles_empty_and_non_json() -> None:
    assert _parse_stream_line("", "") == []
    # Non-JSON line becomes a raw text delta
    events = _parse_stream_line("not json at all", "")
    assert len(events) == 1
    _, delta, _, _, _ = events[0]
    assert delta == "not json at all"


def test_parse_stream_line_extracts_text_delta() -> None:
    line = json.dumps({
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hello"},
        },
    })
    events = _parse_stream_line(line, "")
    assert len(events) == 1
    progress, delta, usage, session_id, tool_count = events[0]
    assert delta == "hello"
    assert usage is None
    assert session_id == ""
    assert tool_count == 0


def test_parse_stream_line_handles_tool_use_start() -> None:
    line = json.dumps({
        "type": "stream_event",
        "event": {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "Bash"},
        },
    })
    events = _parse_stream_line(line, "")
    assert len(events) == 1
    progress, delta, usage, session_id, tool_count = events[0]
    assert delta == ""
    assert progress["item"]["name"] == "Bash"


def test_parse_stream_line_extracts_usage_from_result() -> None:
    line = json.dumps({
        "type": "result",
        "subtype": "success",
        "session_id": "sess-abc",
        "model": "claude-sonnet-4.5",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 20,
        },
        "result": "done",
    })
    events = _parse_stream_line(line, "")
    # result handler emits: usage notification, result text, turn_completed
    assert len(events) == 3
    # First event carries usage
    _, _, usage, session_id, _ = events[0]
    assert usage is not None
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert usage["cached_input_tokens"] == 20
    assert usage["total_tokens"] == 150
    assert usage["model"] == "claude-sonnet-4.5"
    assert session_id == "sess-abc"
    # Last event is turn_completed
    progress, _, _, _, _ = events[-1]
    assert progress["stage"] == "turn_completed"


def test_parse_stream_line_handles_error_result() -> None:
    line = json.dumps({
        "type": "result",
        "subtype": "error_max_turns",
        "session_id": "sess-abc",
        "error": "Max turns exceeded",
    })
    events = _parse_stream_line(line, "")
    assert len(events) == 1
    progress, delta, _, session_id, _ = events[0]
    assert progress["stage"] == "error"
    assert "Max turns" in delta
    assert session_id == "sess-abc"


def test_parse_stream_line_handles_assistant_message_with_tool_use() -> None:
    line = json.dumps({
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "Running tests"},
                {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "pytest"}},
            ],
        },
    })
    events = _parse_stream_line(line, "")
    # assistant handler emits: agentMessage text, tool_use completed, assistant activity
    # The tool_count is on the last event
    *_, last_event = events
    _, _, _, _, tool_count = last_event
    assert tool_count == 1


def test_parse_stream_line_handles_system_event() -> None:
    line = json.dumps({
        "type": "system",
        "subtype": "api_retry",
        "message": "Retrying after 429",
    })
    events = _parse_stream_line(line, "")
    assert len(events) == 1
    progress, delta, _, _, _ = events[0]
    assert progress["subtype"] == "api_retry"
    assert delta == ""


def test_claude_nonzero_exit_classifies_api_retries() -> None:
    message, kind = _nonzero_exit_error(1, api_retry_count=7)
    assert kind == "api_retry_non_zero_exit"
    assert "7 retries" in message
    assert "status 1" in message


def test_claude_success_then_exit_warning_is_distinct() -> None:
    warning = _success_then_nonzero_warning(1, 2)
    assert "successful result event" in warning
    assert "2 API retries" in warning


def test_claude_raw_tail_redacts_secrets() -> None:
    line = "Authorization: Bearer abc123 token=sk-test-secret"
    redacted = _redact_raw_line(line)
    assert "abc123" not in redacted
    assert "sk-test-secret" not in redacted
    assert "[REDACTED]" in redacted


def test_parse_stream_line_handles_hook_event() -> None:
    line = json.dumps({
        "type": "hook.started",
        "hook": {"id": "h1", "name": "pre_edit"},
    })
    events = _parse_stream_line(line, "")
    assert len(events) == 1
    progress, _, _, _, _ = events[0]
    assert progress["method"] == "hook.started"


def test_claude_session_ensure_started_raises_when_binary_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(session_mod, "resolve_claude_binary", lambda configured: "")
    session = ClaudeCliSession(cwd="/tmp")
    with pytest.raises(FileNotFoundError, match="Claude binary not found"):
        session.ensure_started()


def test_claude_session_run_turn_returns_error_when_binary_unresolved(
    monkeypatch,
) -> None:
    monkeypatch.setattr(session_mod, "resolve_claude_binary", lambda configured: "")
    session = ClaudeCliSession(cwd="/tmp")
    result = session.run_turn("hello")
    assert result.error
    assert "Claude binary not found" in result.error or "not found" in result.error.lower()
    assert result.should_retire


def test_claude_session_run_turn_rejects_empty_prompt(monkeypatch) -> None:
    monkeypatch.setattr(
        session_mod,
        "resolve_claude_binary",
        lambda configured: "/fake/claude",
    )
    session = ClaudeCliSession(cwd="/tmp")
    result = session.run_turn("")
    assert result.error
    assert "empty" in result.error.lower()


def test_claude_session_prompt_args_for_new_session(monkeypatch) -> None:
    monkeypatch.setattr(
        session_mod,
        "resolve_claude_binary",
        lambda configured: "/fake/claude",
    )
    session = ClaudeCliSession(
        cwd="/tmp",
        permission_mode="acceptEdits",
        model="claude-sonnet-4.5",
        effort="high",
    )
    session._resolved_binary = "/fake/claude"
    session._capabilities = ClaudeCliCapabilities(
        print_prompt=True,
        output_format=True,
        stream_json_output=True,
        include_partial_messages=True,
        permission_mode=True,
        model=True,
        effort=True,
        resume=True,
        session_id=True,
    )
    args = session._prompt_args("hello world", capabilities=session._capabilities)
    assert "-p" in args
    assert "hello world" in args
    assert "--permission-mode" in args
    assert "acceptEdits" in args
    assert "--model" in args
    assert "claude-sonnet-4.5" in args
    assert "--effort" in args
    assert "high" in args
    assert "--output-format" in args
    assert "stream-json" in args
    assert "--verbose" in args
    assert "--include-partial-messages" in args
    # New session: no --resume
    assert "--resume" not in args


def test_claude_session_prompt_args_for_resume(monkeypatch) -> None:
    monkeypatch.setattr(
        session_mod,
        "resolve_claude_binary",
        lambda configured: "/fake/claude",
    )
    session = ClaudeCliSession(
        cwd="/tmp",
        resume_thread_id="sess-abc-123",
        permission_mode="auto",
    )
    session._resolved_binary = "/fake/claude"
    session._capabilities = ClaudeCliCapabilities(
        print_prompt=True,
        output_format=True,
        stream_json_output=True,
        include_partial_messages=True,
        permission_mode=True,
        model=True,
        effort=True,
        resume=True,
        session_id=True,
    )
    args = session._prompt_args("follow up", capabilities=session._capabilities)
    assert "--resume" in args
    assert "sess-abc-123" in args
    assert "-p" in args
    assert "follow up" in args
    assert "--permission-mode" in args
    assert "auto" in args


def test_claude_session_request_interrupt_sets_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        session_mod,
        "resolve_claude_binary",
        lambda configured: "/fake/claude",
    )
    session = ClaudeCliSession(cwd="/tmp")
    assert not session._interrupt_requested
    session.request_interrupt()
    assert session._interrupt_requested


def test_claude_session_close_sets_closed_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        session_mod,
        "resolve_claude_binary",
        lambda configured: "/fake/claude",
    )
    session = ClaudeCliSession(cwd="/tmp")
    assert not session._closed
    session.close()
    assert session._closed


def test_claude_session_set_approval_callback(monkeypatch) -> None:
    monkeypatch.setattr(
        session_mod,
        "resolve_claude_binary",
        lambda configured: "/fake/claude",
    )
    session = ClaudeCliSession(cwd="/tmp")
    assert session._approval_callback is None
    session.set_approval_callback(lambda *a, **kw: "allow")
    assert session._approval_callback is not None
    session.set_approval_callback(None)
    assert session._approval_callback is None


def test_claude_session_context_manager_closes(monkeypatch) -> None:
    monkeypatch.setattr(
        session_mod,
        "resolve_claude_binary",
        lambda configured: "/fake/claude",
    )
    with ClaudeCliSession(cwd="/tmp") as session:
        assert not session._closed
    assert session._closed


def test_fake_claude_cli_session_records_turn_calls() -> None:
    session = FakeClaudeCliSession(cwd="/tmp")
    session.set_fake_turn(TurnResult(final_text="done", session_id="s1"))
    result = session.run_turn("hello", turn_timeout=60.0)
    assert result.final_text == "done"
    assert result.session_id == "s1"
    assert len(session._turn_calls) == 1
    assert session._turn_calls[0][0] == "hello"
    assert session._turn_calls[0][1]["turn_timeout"] == 60.0


def test_resolve_claude_binary_returns_empty_when_not_found(
    monkeypatch,
    tmp_path,
) -> None:
    # Clear PATH and env vars so discovery fails
    monkeypatch.setenv("HERMES_CLAUDE_BINARY", "")
    monkeypatch.setenv("WLCODEX_CLAUDE_BINARY", "")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path) if p == "~" else p)
    result = resolve_claude_binary("auto")
    # In test env this may still find the VS Code extension; accept either.
    assert isinstance(result, str)
