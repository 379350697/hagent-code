"""Session adapter for the Claude Agent SDK runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from collections import deque
from contextlib import suppress
from typing import Any, Callable, Optional

from agent.transports.claude_runtime import TurnResult

logger = logging.getLogger(__name__)

_RESULT_GRACE_AFTER_ASSISTANT_END_TURN_SECONDS = 2.0
_GATEWAY_LIFECYCLE_COMMAND_RE = re.compile(
    r"("
    r"\bsystemctl\s+--user\s+(restart|stop)\s+hermes-gateway\.service\b"
    r"|\bhermes\s+gateway\s+(restart|stop)\b"
    r"|\bhermes\s+update\b"
    r")",
    re.IGNORECASE,
)
_HERMES_RUNTIME_POLICY = (
    "Hermes runtime policy: do not restart, stop, or update the Hermes gateway "
    "from inside this Claude turn. Gateway lifecycle operations terminate the "
    "process that is carrying this conversation. If deployment verification "
    "requires a gateway restart, finish all edits, tests, and the final answer "
    "first, then ask the operator or Hermes host to perform the restart as the "
    "last external step."
)


class ClaudeAgentSdkRunner:
    """Thin wrapper around claude-agent-sdk for testable execution."""

    async def run(
        self,
        *,
        prompt: str,
        cwd: str,
        session_id: str,
        config: dict[str, Any],
        permission_mode: str,
        model: str,
        effort: str,
        interrupt_state: dict[str, Any],
    ):
        try:
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
        except ImportError as exc:
            raise RuntimeError("claude-agent-sdk is not installed") from exc

        env = _sdk_env(config)
        session_key = session_id or "default"

        async def prompt_stream():
            yield {
                "type": "user",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
                "session_id": session_key,
            }

        system_prompt = str(config.get("system_prompt") or "").strip()
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{_HERMES_RUNTIME_POLICY}"
        else:
            system_prompt = _HERMES_RUNTIME_POLICY

        options = ClaudeAgentOptions(
            cwd=cwd,
            model=model or config.get("model") or None,
            effort=effort or config.get("effort") or None,
            system_prompt=system_prompt,
            permission_mode=permission_mode or None,
            resume=session_id or None,
            cli_path=config.get("cli_path") or None,
            env=env,
            include_partial_messages=True,
            include_hook_events=True,
            can_use_tool=_hermes_can_use_tool,
        )
        client = ClaudeSDKClient(options=options)
        try:
            await client.connect(prompt_stream())
            interrupt_state["client"] = client
            interrupt_state["loop"] = asyncio.get_running_loop()
            async for message in client.receive_response():
                yield message
        finally:
            await client.disconnect()


class ClaudeAgentSdkSession:
    """Runs Hermes Claude turns through Python claude-agent-sdk."""

    def __init__(
        self,
        *,
        cwd: str,
        config_overrides: Optional[list[str]] = None,
        resume_thread_id: str = "",
        permission_mode: str = "acceptEdits",
        model: str = "",
        effort: str = "",
        sdk_profile_config: Optional[dict[str, Any]] = None,
        runner: Optional[Any] = None,
        **_: Any,
    ) -> None:
        self.cwd = cwd
        self.config_overrides = list(config_overrides or [])
        self.resume_thread_id = str(resume_thread_id or "")
        self.thread_id = self.resume_thread_id
        self.session_id = self.resume_thread_id
        self._permission_mode = permission_mode or "acceptEdits"
        self._model = model or ""
        self._effort = effort or ""
        self._config = dict(sdk_profile_config or {})
        self._runner = runner or ClaudeAgentSdkRunner()
        self._approval_callback: Optional[Callable[..., str]] = None
        self._closed = False
        self._interrupt_requested = False
        self._interrupt_state: dict[str, Any] = {}
        self._started_once = False

    def ensure_started(self) -> str:
        error = _profile_preflight_error(self._config)
        if error:
            raise RuntimeError(error)
        if self._runner.__class__ is ClaudeAgentSdkRunner:
            try:
                import claude_agent_sdk  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("claude-agent-sdk is not installed") from exc
        return self.thread_id

    def set_approval_callback(
        self,
        callback: Optional[Callable[..., str]],
    ) -> None:
        self._approval_callback = callback

    def close(self) -> None:
        self._closed = True
        self._interrupt_state.clear()

    def request_interrupt(self) -> None:
        self._interrupt_requested = True
        client = self._interrupt_state.get("client")
        loop = self._interrupt_state.get("loop")
        if client is None or loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(client.interrupt(), loop)
        except Exception:
            logger.debug("Claude SDK interrupt scheduling failed", exc_info=True)

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: float = 1800.0,
        idle_timeout: float = 600.0,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        **_: Any,
    ) -> TurnResult:
        prompt_text = _coerce_turn_input_text(user_input)
        if not prompt_text:
            return TurnResult(
                error="Claude SDK turn rejected: empty prompt.",
                error_kind="empty_prompt",
                runtime="agent_sdk",
                runtime_profile=str(self._config.get("name") or ""),
            )
        try:
            self.ensure_started()
        except Exception as exc:
            return TurnResult(
                error=str(exc),
                error_kind="sdk_preflight_failed",
                should_retire=True,
                runtime="agent_sdk",
                runtime_profile=str(self._config.get("name") or ""),
                started=False,
            )
        self._interrupt_requested = False
        try:
            return asyncio.run(
                self._run_turn_async(
                    prompt_text,
                    turn_timeout=turn_timeout,
                    idle_timeout=idle_timeout,
                    progress_callback=progress_callback,
                )
            )
        except RuntimeError:
            return self._run_turn_in_thread(
                prompt_text,
                turn_timeout=turn_timeout,
                idle_timeout=idle_timeout,
                progress_callback=progress_callback,
            )

    async def _run_turn_async(
        self,
        prompt: str,
        *,
        turn_timeout: float,
        idle_timeout: float,
        progress_callback: Optional[Callable[[dict[str, Any]], None]],
    ) -> TurnResult:
        started_at = time.monotonic()
        last_activity_at = started_at
        assistant_text = ""
        accumulated: list[str] = []
        tool_iterations = 0
        token_usage: Optional[dict[str, Any]] = None
        result_message: Any = None
        terminal_assistant: _TerminalAssistant | None = None
        terminal_assistant_deadline: float | None = None
        raw_tail: deque[str] = deque(maxlen=50)
        last_session_id = self.session_id
        self._started_once = True
        if progress_callback is not None:
            progress_callback({"stage": "turn_started", "runtime": "agent_sdk"})

        try:
            async_iter = self._runner.run(
                prompt=prompt,
                cwd=self.cwd,
                session_id=self.session_id,
                config=self._config,
                permission_mode=self._permission_mode,
                model=self._model,
                effort=self._effort,
                interrupt_state=self._interrupt_state,
            ).__aiter__()
            pending_next: asyncio.Task[Any] | None = None
            while True:
                if self._interrupt_requested:
                    if pending_next is not None and not pending_next.done():
                        pending_next.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            await pending_next
                    return TurnResult(
                        final_text="".join(accumulated),
                        interrupted=True,
                        error_kind="interrupted",
                        session_id=last_session_id,
                        token_usage_last=token_usage,
                        token_usage_total=token_usage,
                        tool_iterations=tool_iterations,
                        raw_output_tail=list(raw_tail),
                        runtime="agent_sdk",
                        runtime_profile=str(self._config.get("name") or ""),
                        started=True,
                    )
                timeout = _next_timeout(
                    started_at=started_at,
                    last_activity_at=last_activity_at,
                    turn_timeout=turn_timeout,
                    idle_timeout=idle_timeout,
                )
                if pending_next is None:
                    pending_next = asyncio.create_task(async_iter.__anext__())
                try:
                    message = await asyncio.wait_for(asyncio.shield(pending_next), timeout=timeout)
                    pending_next = None
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if (
                        terminal_assistant is not None
                        and terminal_assistant_deadline is not None
                        and now >= terminal_assistant_deadline
                    ):
                        if pending_next is not None and not pending_next.done():
                            pending_next.cancel()
                            with suppress(asyncio.CancelledError, Exception):
                                await pending_next
                        break
                    hard_expired = turn_timeout > 0 and now - started_at >= turn_timeout
                    idle_expired = idle_timeout > 0 and now - last_activity_at >= idle_timeout
                    if not hard_expired and not idle_expired:
                        continue
                    if pending_next is not None and not pending_next.done():
                        pending_next.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            await pending_next
                    reason = "hard" if hard_expired else "idle"
                    return TurnResult(
                        final_text="".join(accumulated),
                        error=(
                            f"Claude SDK turn timed out after {turn_timeout:g}s ({reason})"
                            if reason == "hard"
                            else f"Claude SDK idle-timed out after {idle_timeout:g}s."
                        ),
                        error_kind=f"sdk_timeout_{reason}",
                        should_retire=True,
                        session_id=last_session_id,
                        token_usage_last=token_usage,
                        token_usage_total=token_usage,
                        tool_iterations=tool_iterations,
                        raw_output_tail=list(raw_tail),
                        runtime="agent_sdk",
                        runtime_profile=str(self._config.get("name") or ""),
                        started=True,
                    )
                last_activity_at = time.monotonic()
                raw_tail.append(_redact_raw_line(_message_summary(message)))
                parsed = _parse_sdk_message(message, assistant_text)
                if parsed.session_id:
                    last_session_id = parsed.session_id
                if parsed.usage:
                    token_usage = parsed.usage
                if parsed.tool_count:
                    tool_iterations += parsed.tool_count
                if parsed.delta_text:
                    accumulated.append(parsed.delta_text)
                    assistant_text += parsed.delta_text
                if parsed.result_message is not None:
                    result_message = parsed.result_message
                    break
                if parsed.terminal_assistant is not None:
                    terminal_assistant = parsed.terminal_assistant
                    terminal_assistant_deadline = (
                        time.monotonic() + _RESULT_GRACE_AFTER_ASSISTANT_END_TURN_SECONDS
                    )
                if parsed.error:
                    return TurnResult(
                        final_text="".join(accumulated),
                        error=parsed.error,
                        error_kind=parsed.error_kind or "sdk_assistant_error",
                        should_retire=True,
                        session_id=last_session_id,
                        token_usage_last=token_usage,
                        token_usage_total=token_usage,
                        tool_iterations=tool_iterations,
                        raw_output_tail=list(raw_tail),
                        runtime="agent_sdk",
                        runtime_profile=str(self._config.get("name") or ""),
                        started=True,
                    )
                if progress_callback is not None:
                    for event in parsed.progress_events:
                        progress_callback(event)
        except Exception as exc:
            return TurnResult(
                final_text="".join(accumulated),
                error=str(exc),
                error_kind=_sdk_exception_kind(exc),
                should_retire=not self._started_once,
                session_id=last_session_id,
                token_usage_last=token_usage,
                token_usage_total=token_usage,
                tool_iterations=tool_iterations,
                raw_output_tail=list(raw_tail),
                runtime="agent_sdk",
                runtime_profile=str(self._config.get("name") or ""),
                started=self._started_once,
            )
        finally:
            self._interrupt_state.clear()

        result = _result_from_sdk_message(
            result_message,
            fallback_text="".join(accumulated),
            terminal_assistant=terminal_assistant,
        )
        if result.session_id:
            last_session_id = result.session_id
        if result.token_usage_total:
            token_usage = result.token_usage_total
        if last_session_id:
            self.session_id = last_session_id
            self.thread_id = last_session_id
        result.final_text = result.final_text or "".join(accumulated)
        result.session_id = last_session_id
        result.token_usage_last = result.token_usage_last or token_usage
        result.token_usage_total = result.token_usage_total or token_usage
        result.tool_iterations = max(result.tool_iterations, tool_iterations)
        result.raw_output_tail = list(raw_tail)
        result.runtime = "agent_sdk"
        result.runtime_profile = str(self._config.get("name") or "")
        result.started = True
        return result

    def _run_turn_in_thread(
        self,
        prompt: str,
        *,
        turn_timeout: float,
        idle_timeout: float,
        progress_callback: Optional[Callable[[dict[str, Any]], None]],
    ) -> TurnResult:
        result_box: dict[str, object] = {}

        def worker() -> None:
            try:
                result_box["value"] = asyncio.run(
                    self._run_turn_async(
                        prompt,
                        turn_timeout=turn_timeout,
                        idle_timeout=idle_timeout,
                        progress_callback=progress_callback,
                    )
                )
            except BaseException as exc:
                result_box["error"] = exc

        thread = threading.Thread(target=worker, name="claude-sdk-turn-worker", daemon=True)
        thread.start()
        while thread.is_alive():
            thread.join(timeout=0.1)
        if "error" in result_box:
            raise result_box["error"]  # type: ignore[misc]
        return result_box.get("value")  # type: ignore[return-value]


async def _hermes_can_use_tool(tool_name: str, tool_input: dict[str, Any], _context: Any):
    try:
        from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny
    except Exception:  # pragma: no cover - only used with the SDK installed.
        return None

    command = _tool_command(tool_name, tool_input)
    if command and _is_gateway_lifecycle_command(command):
        return PermissionResultDeny(
            message=(
                "Hermes blocked this gateway lifecycle command because it would "
                "terminate the running Claude turn. Finish the task and final "
                "response first; perform the gateway restart as the last external step."
            ),
            interrupt=False,
        )
    return PermissionResultAllow()


def _tool_command(tool_name: str, tool_input: dict[str, Any]) -> str:
    if str(tool_name or "").lower() != "bash":
        return ""
    command = tool_input.get("command") if isinstance(tool_input, dict) else ""
    return command if isinstance(command, str) else ""


def _is_gateway_lifecycle_command(command: str) -> bool:
    return bool(_GATEWAY_LIFECYCLE_COMMAND_RE.search(command or ""))


class _ParsedSdkMessage:
    def __init__(
        self,
        *,
        delta_text: str = "",
        session_id: str = "",
        usage: Optional[dict[str, Any]] = None,
        tool_count: int = 0,
        result_message: Any = None,
        terminal_assistant: Optional["_TerminalAssistant"] = None,
        error: str = "",
        error_kind: str = "",
        progress_events: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.delta_text = delta_text
        self.session_id = session_id
        self.usage = usage
        self.tool_count = tool_count
        self.result_message = result_message
        self.terminal_assistant = terminal_assistant
        self.error = error
        self.error_kind = error_kind
        self.progress_events = list(progress_events or [])


class _TerminalAssistant:
    def __init__(
        self,
        *,
        text: str,
        session_id: str = "",
        usage: Optional[dict[str, Any]] = None,
        stop_reason: str = "",
    ) -> None:
        self.text = text
        self.session_id = session_id
        self.usage = usage
        self.stop_reason = stop_reason


def _parse_sdk_message(message: Any, assistant_text: str) -> _ParsedSdkMessage:
    class_name = message.__class__.__name__
    if class_name == "SystemMessage":
        data = getattr(message, "data", {}) or {}
        session_id = str(data.get("session_id") or "")
        subtype = str(getattr(message, "subtype", "") or "")
        if subtype == "thinking_tokens":
            return _ParsedSdkMessage(session_id=session_id)
        return _ParsedSdkMessage(
            session_id=session_id,
            progress_events=[{
                "stage": "notification",
                "method": "system",
                "subtype": subtype,
                "runtime": "agent_sdk",
            }],
        )
    if class_name == "AssistantMessage":
        text, tool_events, tool_count = _assistant_parts(message)
        delta = text[len(assistant_text):] if text and text.startswith(assistant_text) else text
        usage = _extract_usage(getattr(message, "usage", None))
        session_id = str(getattr(message, "session_id", "") or "")
        stop_reason = str(getattr(message, "stop_reason", "") or "")
        error = getattr(message, "error", None)
        if error:
            return _ParsedSdkMessage(
                delta_text=delta,
                session_id=session_id,
                usage=usage,
                tool_count=tool_count,
                error=str(error),
                error_kind="sdk_assistant_error",
            )
        events = []
        if delta:
            events.append({
                "stage": "notification",
                "method": "item/completed",
                "item": {"type": "agentMessage"},
                "runtime": "agent_sdk",
            })
        events.extend(tool_events)
        return _ParsedSdkMessage(
            delta_text=delta,
            session_id=session_id,
            usage=usage,
            tool_count=tool_count,
            terminal_assistant=(
                _TerminalAssistant(
                    text=text,
                    session_id=session_id,
                    usage=usage,
                    stop_reason=stop_reason,
                )
                if stop_reason == "end_turn" and text.strip()
                else None
            ),
            progress_events=events,
        )
    if class_name == "ResultMessage":
        session_id = str(getattr(message, "session_id", "") or "")
        usage = _extract_usage(getattr(message, "usage", None))
        events = []
        if usage:
            events.append({
                "stage": "notification",
                "method": "thread/tokenUsage/updated",
                "usage": usage,
                "runtime": "agent_sdk",
            })
        events.append({
            "stage": "turn_completed",
            "subtype": str(getattr(message, "subtype", "") or ""),
            "session_id": session_id,
            "runtime": "agent_sdk",
        })
        return _ParsedSdkMessage(
            session_id=session_id,
            usage=usage,
            result_message=message,
            progress_events=events,
        )
    if class_name == "RateLimitEvent":
        return _ParsedSdkMessage(
            progress_events=[{
                "stage": "notification",
                "method": "rate_limit",
                "runtime": "agent_sdk",
            }],
        )
    if class_name == "StreamEvent":
        event = getattr(message, "event", {}) or {}
        delta = event.get("delta") if isinstance(event, dict) else {}
        if isinstance(delta, dict) and delta.get("type") == "thinking_delta":
            return _ParsedSdkMessage(
                session_id=str(getattr(message, "session_id", "") or "")
            )
        return _ParsedSdkMessage(
            session_id=str(getattr(message, "session_id", "") or ""),
            progress_events=[{
                "stage": "notification",
                "method": "StreamEvent",
                "runtime": "agent_sdk",
            }],
        )
    return _ParsedSdkMessage(
        progress_events=[{
            "stage": "notification",
            "method": class_name or "sdk_message",
            "runtime": "agent_sdk",
        }],
    )


def _assistant_parts(message: Any) -> tuple[str, list[dict[str, Any]], int]:
    content = getattr(message, "content", []) or []
    text_parts: list[str] = []
    events: list[dict[str, Any]] = []
    tool_count = 0
    for block in content:
        block_type = getattr(block, "type", "") or (
            block.get("type") if isinstance(block, dict) else ""
        )
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str):
            text_parts.append(text)
        name = getattr(block, "name", None)
        if name is None and isinstance(block, dict):
            name = block.get("name")
        if name or block_type == "tool_use" or block.__class__.__name__ == "ToolUseBlock":
            tool_count += 1
            events.append({
                "stage": "notification",
                "method": "item/completed",
                "item": {"type": "tool_use", "name": str(name or "unknown")},
                "runtime": "agent_sdk",
            })
    return "".join(text_parts), events, tool_count


def _result_from_sdk_message(
    message: Any,
    *,
    fallback_text: str,
    terminal_assistant: Optional[_TerminalAssistant] = None,
) -> TurnResult:
    if message is None:
        if terminal_assistant is not None:
            return TurnResult(
                final_text=terminal_assistant.text or fallback_text,
                session_id=terminal_assistant.session_id,
                token_usage_last=terminal_assistant.usage,
                token_usage_total=terminal_assistant.usage,
                warning="Claude SDK finished without ResultMessage.",
                error_kind="sdk_missing_result",
            )
        return TurnResult(
            final_text=fallback_text,
            error="Claude SDK ended without a ResultMessage.",
            error_kind="sdk_missing_result",
            should_retire=True,
        )
    subtype = str(getattr(message, "subtype", "") or "")
    is_error = bool(getattr(message, "is_error", False))
    result_text = str(getattr(message, "result", "") or "")
    errors = getattr(message, "errors", None)
    usage = _extract_usage(getattr(message, "usage", None))
    session_id = str(getattr(message, "session_id", "") or "")
    if subtype == "success" and not is_error:
        return TurnResult(
            final_text=result_text or fallback_text,
            session_id=session_id,
            token_usage_last=usage,
            token_usage_total=usage,
        )
    error_text = result_text or "; ".join(str(item) for item in (errors or [])) or subtype or "Claude SDK failed."
    return TurnResult(
        final_text=fallback_text,
        error=error_text,
        error_kind=_sdk_result_error_kind(message),
        should_retire=True,
        session_id=session_id,
        token_usage_last=usage,
        token_usage_total=usage,
    )


def _sdk_result_error_kind(message: Any) -> str:
    status = getattr(message, "api_error_status", None)
    if status == 429:
        return "sdk_rate_limit"
    if status in {401, 403}:
        return "sdk_auth_error"
    subtype = str(getattr(message, "subtype", "") or "")
    return f"sdk_{subtype}" if subtype else "sdk_error"


def _sdk_exception_kind(exc: BaseException) -> str:
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    if "not installed" in text or "notfound" in name:
        return "sdk_dependency_missing"
    if "auth" in text or "401" in text or "403" in text:
        return "sdk_auth_error"
    if "rate" in text or "429" in text:
        return "sdk_rate_limit"
    if "json" in name:
        return "sdk_json_decode"
    if "process" in name:
        return "sdk_process_error"
    if "connection" in name or "connect" in text:
        return "sdk_connection_error"
    return "sdk_exception"


def _extract_usage(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    for src_key, dst_key in [
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("cache_read_input_tokens", "cached_input_tokens"),
        ("cache_creation_input_tokens", "cached_input_tokens"),
    ]:
        val = value.get(src_key)
        if isinstance(val, (int, float)):
            result[dst_key] = result.get(dst_key, 0) + int(val)
    if "input_tokens" in result and "output_tokens" in result:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
        result["source"] = "exact"
        return result
    return None


def _profile_preflight_error(config: dict[str, Any]) -> str:
    profile = str(config.get("name") or "unknown")
    api_key = str(config.get("api_key") or "")
    api_key_env = str(config.get("api_key_env") or "")
    if not api_key and api_key_env and os.environ.get(api_key_env):
        return ""
    if not api_key:
        return f"Claude SDK profile {profile} has no API key."
    return ""


def _sdk_env(config: dict[str, Any]) -> dict[str, str]:
    api_key = str(config.get("api_key") or "")
    api_key_env = str(config.get("api_key_env") or "")
    if not api_key and api_key_env:
        api_key = os.environ.get(api_key_env, "")
    env: dict[str, str] = {}
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
        env["ANTHROPIC_AUTH_TOKEN"] = api_key
    base_url = str(config.get("base_url") or "")
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url
    for key, value in (config.get("extra_env") or {}).items():
        if value is not None:
            env[str(key)] = str(value)
    return env


def _coerce_turn_input_text(user_input: Any) -> str:
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        parts: list[str] = []
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "input_text"}:
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                elif item.get("type") in {"image", "image_url", "input_image"}:
                    parts.append("[image attached]")
        return "\n\n".join(part for part in parts if part).strip()
    return "" if user_input is None else str(user_input)


def _next_timeout(
    *,
    started_at: float,
    last_activity_at: float,
    turn_timeout: float,
    idle_timeout: float,
) -> float:
    timeout = 0.5
    now = time.monotonic()
    if turn_timeout > 0:
        timeout = min(timeout, max(0.001, turn_timeout - (now - started_at)))
    if idle_timeout > 0:
        timeout = min(timeout, max(0.001, idle_timeout - (now - last_activity_at)))
    return timeout


def _message_summary(message: Any) -> str:
    try:
        return json.dumps(_safe_message_summary(message), ensure_ascii=False, sort_keys=True)
    except Exception:
        return message.__class__.__name__ or "sdk_message"


def _safe_message_summary(message: Any) -> dict[str, Any]:
    class_name = message.__class__.__name__
    if class_name == "SystemMessage":
        data = getattr(message, "data", {}) or {}
        return {
            "type": class_name,
            "subtype": str(getattr(message, "subtype", "") or ""),
            "session_id": str(data.get("session_id") or ""),
            "estimated_tokens": data.get("estimated_tokens"),
            "estimated_tokens_delta": data.get("estimated_tokens_delta"),
        }
    if class_name == "AssistantMessage":
        content = getattr(message, "content", []) or []
        tool_names: list[str] = []
        block_types: list[str] = []
        for block in content:
            block_type = getattr(block, "type", "") or (
                block.get("type") if isinstance(block, dict) else ""
            )
            if block_type:
                block_types.append(str(block_type))
            name = getattr(block, "name", None)
            if name is None and isinstance(block, dict):
                name = block.get("name")
            if name:
                tool_names.append(str(name))
        return {
            "type": class_name,
            "session_id": str(getattr(message, "session_id", "") or ""),
            "stop_reason": str(getattr(message, "stop_reason", "") or ""),
            "error": str(getattr(message, "error", "") or ""),
            "content_types": block_types[:20],
            "tool_names": tool_names[:20],
            "usage": _extract_usage(getattr(message, "usage", None)),
        }
    if class_name == "ResultMessage":
        return {
            "type": class_name,
            "session_id": str(getattr(message, "session_id", "") or ""),
            "subtype": str(getattr(message, "subtype", "") or ""),
            "is_error": bool(getattr(message, "is_error", False)),
            "stop_reason": str(getattr(message, "stop_reason", "") or ""),
            "api_error_status": getattr(message, "api_error_status", None),
            "usage": _extract_usage(getattr(message, "usage", None)),
        }
    if class_name == "StreamEvent":
        event = getattr(message, "event", {}) or {}
        delta = event.get("delta") if isinstance(event, dict) else {}
        content_block = event.get("content_block") if isinstance(event, dict) else {}
        usage = event.get("usage") if isinstance(event, dict) else None
        return {
            "type": class_name,
            "session_id": str(getattr(message, "session_id", "") or ""),
            "event_type": str(event.get("type") or "") if isinstance(event, dict) else "",
            "delta_type": str(delta.get("type") or "") if isinstance(delta, dict) else "",
            "content_block_type": (
                str(content_block.get("type") or "") if isinstance(content_block, dict) else ""
            ),
            "stop_reason": (
                str((event.get("delta") or {}).get("stop_reason") or "")
                if isinstance(event, dict) and isinstance(event.get("delta"), dict)
                else ""
            ),
            "usage": _extract_usage(usage),
        }
    return {"type": class_name or "sdk_message"}


def _to_safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_safe_json(item) for item in value[:20]]
    if hasattr(value, "__dict__"):
        return {
            "type": value.__class__.__name__,
            **{
                str(k): _to_safe_json(v)
                for k, v in vars(value).items()
                if not k.startswith("_")
            },
        }
    return repr(value)


_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b("
    r"(?:[A-Za-z0-9_]*_)?"
    r"(?:password|passwd|token|secret|api[_-]?key|access[_-]?key|private[_-]?key|credential|authorization|auth)"
    r"(?:_[A-Za-z0-9_]*)?"
    r")\s*([=:])\s*([\"']?)([^\s\"'`;,]+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\b(Bearer)\s+([A-Za-z0-9._~+/\-]+=*)", re.IGNORECASE)
_AUTH_HEADER_RE = re.compile(
    r"\b(Authorization)\s*:\s*(?:Bearer\s+)?([^\s;,]+)",
    re.IGNORECASE,
)
_SK_SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{6,}\b")


def _redact_raw_line(text: str) -> str:
    redacted = _AUTH_HEADER_RE.sub(lambda m: f"{m.group(1)}: [REDACTED]", text)
    redacted = _SECRET_ASSIGNMENT_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}[REDACTED]",
        redacted,
    )
    redacted = _BEARER_RE.sub(lambda m: f"{m.group(1)} [REDACTED]", redacted)
    redacted = _SK_SECRET_RE.sub("[REDACTED]", redacted)
    return redacted[:2000]
