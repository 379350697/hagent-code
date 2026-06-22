"""Session adapter for the Claude Code CLI runtime.

Owns one Claude conversation per Hermes session. Each ``run_turn`` spawns a
``claude -p`` subprocess with ``--output-format stream-json --verbose``, parses
the stream-json lines into progress + final text, and returns a clean
``TurnResult`` that mirrors :class:`CodexAppServerSession.run_turn`.

Lifecycle:
    session = ClaudeCliSession(cwd="/home/x/proj")
    session.ensure_started()                       # resolves binary + probes capabilities
    result = session.run_turn(user_input="hello")  # blocks until subprocess exits
    # result.final_text          → assistant text returned to caller
    # result.tool_iterations     → how many tool_use blocks completed
    # result.interrupted         → True if the subprocess was interrupted
    # result.session_id          → Claude session id (from result event)
    session.close()                                # no-op for subprocess transport

Threading model: synchronous from the caller's perspective. The blocking
subprocess is driven by ``run_turn`` which polls the reader thread owned by
``asyncio.subprocess.Process``. Callers should dispatch via
``run_blocking`` so the adapter event loop is not blocked.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from agent.transports.claude_runtime import TurnResult

logger = logging.getLogger(__name__)


# Default binary discovery mirrors wlcodex/claude_binary.py:
# 1. WLCODEX_CLAUDE_BINARY / HERMES_CLAUDE_BINARY env var
# 2. configured value (if not "auto")
# 3. PATH lookup for `claude`
# 4. ~/.local/bin/claude
# 5. newest VS Code extension native-binary/claude
def resolve_claude_binary(configured: str = "auto") -> str:
    env_binary = (
        os.environ.get("HERMES_CLAUDE_BINARY")
        or os.environ.get("WLCODEX_CLAUDE_BINARY")
        or ""
    ).strip()
    if env_binary:
        path = Path(env_binary).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)

    configured = (configured or "auto").strip()
    if configured and configured.lower() != "auto":
        path = Path(configured).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)

    import shutil

    resolved = shutil.which("claude")
    if resolved:
        return resolved

    local_bin = Path.home() / ".local" / "bin" / "claude"
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return str(local_bin)

    # VS Code extension fallback
    extension_dir = Path.home() / ".vscode" / "extensions"
    candidates = [
        path
        for path in extension_dir.glob(
            "anthropic.claude-code-*/resources/native-binary/claude"
        )
        if path.is_file() and os.access(path, os.X_OK)
    ]
    if candidates:
        import re

        def _version_key(path: Path) -> tuple[int, ...]:
            match = re.search(
                r"anthropic\.claude-code-([0-9]+(?:\.[0-9]+)*)", str(path)
            )
            if not match:
                return (0,)
            return tuple(int(part) for part in match.group(1).split("."))

        return str(sorted(candidates, key=_version_key)[-1])

    return ""


@dataclass
class ClaudeCliCapabilities:
    print_prompt: bool = False
    output_format: bool = False
    stream_json_output: bool = False
    include_partial_messages: bool = False
    include_hook_events: bool = False
    permission_mode: bool = False
    model: bool = False
    effort: bool = False
    resume: bool = False
    session_id: bool = False
    probe_error: str = ""

    @classmethod
    def minimal(cls) -> "ClaudeCliCapabilities":
        return cls(print_prompt=True)


def parse_claude_help(help_text: str) -> ClaudeCliCapabilities:
    return ClaudeCliCapabilities(
        print_prompt="-p, --print" in help_text or "--print" in help_text,
        output_format="--output-format" in help_text,
        stream_json_output=(
            "--output-format" in help_text and "stream-json" in help_text
        ),
        include_partial_messages="--include-partial-messages" in help_text,
        include_hook_events="--include-hook-events" in help_text,
        permission_mode="--permission-mode" in help_text,
        model="--model" in help_text,
        effort="--effort" in help_text,
        resume="--resume" in help_text or "-r, --resume" in help_text,
        session_id="--session-id" in help_text,
    )


async def probe_claude_capabilities(
    binary: str,
    *,
    timeout_seconds: float = 5.0,
) -> ClaudeCliCapabilities:
    if not binary:
        return ClaudeCliCapabilities(probe_error="binary_not_found")
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return ClaudeCliCapabilities(probe_error="binary_not_found")
    except Exception:
        return ClaudeCliCapabilities.minimal()

    text = ""
    if stdout:
        text += stdout.decode("utf-8", errors="replace")
    if stderr:
        text += "\n" + stderr.decode("utf-8", errors="replace")
    caps = parse_claude_help(text)
    return caps if caps.print_prompt else ClaudeCliCapabilities.minimal()


@dataclass(frozen=True)
class ClaudeTurnOptions:
    """Per-turn Claude CLI options.

    Mirrors the subset of :func:`codex_app_server_turn_options` that applies to
    Claude. Timeouts are configurable from Hermes config so long-running remote
    Claude tasks do not trip the transport default.
    """

    turn_timeout: float = 1800.0
    idle_timeout: float = 600.0
    drain_grace_seconds: float = 0.1


def default_turn_options() -> ClaudeTurnOptions:
    return ClaudeTurnOptions()


class ClaudeCliSession:
    """Owns one Claude conversation per Hermes session.

    Unlike :class:`CodexAppServerSession`, there is no long-lived app-server
    subprocess. Each :meth:`run_turn` spawns a fresh ``claude -p`` subprocess
    that exits when the turn completes. Continuation across turns is done via
    ``--resume <session_id>`` which Claude resolves from its local session
    store under ``~/.claude/projects/``.
    """

    def __init__(
        self,
        *,
        cwd: str,
        config_overrides: Optional[list[str]] = None,
        resume_thread_id: str = "",
        binary: str = "auto",
        permission_mode: str = "acceptEdits",
        model: str = "",
        effort: str = "",
        session_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.cwd = cwd
        self.config_overrides = list(config_overrides or [])
        self.resume_thread_id = str(resume_thread_id or "")
        self._binary_setting = binary or "auto"
        self._permission_mode = permission_mode or "acceptEdits"
        self._model = model or ""
        self._effort = effort or ""
        self._session_factory = session_factory
        self._capabilities: Optional[ClaudeCliCapabilities] = None
        self._resolved_binary: str = ""
        self._binary_resolution_error: str = ""
        self.thread_id: str = self.resume_thread_id
        self.session_id: str = self.resume_thread_id
        self._approval_callback: Optional[Callable[..., str]] = None
        self._closed = False
        self._interrupt_requested = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_started(self) -> str:
        """Resolve the binary and probe capabilities.

        Returns the thread/session id that this session will resume from.
        For a new session this is empty; the first :meth:`run_turn` will
        populate ``self.session_id`` from the Claude result event.
        """
        if self._resolved_binary:
            return self.thread_id
        self._resolved_binary = resolve_claude_binary(self._binary_setting)
        if not self._resolved_binary:
            self._binary_resolution_error = (
                "Claude binary not found. Set HERMES_CLAUDE_BINARY or install "
                "Claude Code CLI (`npm install -g @anthropic-ai/claude-code`)."
            )
            raise FileNotFoundError(self._binary_resolution_error)
        return self.thread_id

    def set_approval_callback(
        self, callback: Optional[Callable[..., str]]
    ) -> None:
        self._approval_callback = callback

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "ClaudeCliSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def request_interrupt(self) -> None:
        """Request interruption of the in-flight turn (if any).

        The next :meth:`run_turn` iteration will observe this flag and kill the
        subprocess. Because each turn owns its own subprocess, there is no
        long-lived process to interrupt between turns.
        """
        self._interrupt_requested = True

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: float = 1800.0,
        idle_timeout: float = 600.0,
        post_tool_quiet_timeout: float = 90.0,
        active_tool_timeout: float = 3600.0,
        notification_poll_timeout: float = 0.25,
        unbounded_command_policy: str = "conditional_hard",
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> TurnResult:
        """Run one Claude CLI turn synchronously.

        Spawns ``claude -p <prompt> --output-format stream-json --verbose`` and
        blocks until the subprocess exits. Returns a :class:`TurnResult` with
        the final assistant text, token usage, and session id.
        """
        del post_tool_quiet_timeout, active_tool_timeout
        del notification_poll_timeout, unbounded_command_policy

        if self._binary_resolution_error:
            return TurnResult(error=self._binary_resolution_error, should_retire=True)
        if not self._resolved_binary:
            try:
                self.ensure_started()
            except FileNotFoundError as exc:
                return TurnResult(error=str(exc), should_retire=True)

        prompt_text = _coerce_turn_input_text(user_input)
        if not prompt_text:
            return TurnResult(error="Claude turn rejected: empty prompt.")

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
            # asyncio.run cannot be called from a running loop — fall back to
            # a dedicated thread bridge (used by the gateway adapter).
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
        capabilities = await self._probe_capabilities()
        args = self._prompt_args(prompt, capabilities=capabilities)
        cwd = self.cwd or None

        try:
            proc = await asyncio.create_subprocess_exec(
                self._resolved_binary,
                *args,
                cwd=cwd,
                env=_sanitized_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                limit=1024 * 1024 * 16,
                start_new_session=True,
            )
        except FileNotFoundError:
            return TurnResult(
                error=f"Claude binary not found: {self._resolved_binary}",
                should_retire=True,
            )

        if proc.stdout is None:
            return TurnResult(error="(no stdout from Claude)", should_retire=True)

        accumulated: list[str] = []
        assistant_text = ""
        tool_iterations = 0
        last_session_id = self.session_id
        token_usage: Optional[dict[str, Any]] = None
        stream_error = ""
        stream_error_kind = ""
        api_retry_count = 0
        result_success_seen = False
        raw_tail: deque[str] = deque(maxlen=50)
        reader_task: Optional[asyncio.Task[bytes]] = asyncio.create_task(
            proc.stdout.readline()
        )
        wait_task: Optional[asyncio.Task[int]] = asyncio.create_task(proc.wait())
        started_at = asyncio.get_running_loop().time()
        last_activity_at = started_at
        timeout_reason = ""
        process_exited = False
        stdout_eof = False

        try:
            while True:
                now = asyncio.get_running_loop().time()
                if turn_timeout > 0 and now - started_at >= turn_timeout:
                    timeout_reason = "hard"
                    break
                if idle_timeout > 0 and now - last_activity_at >= idle_timeout:
                    timeout_reason = "idle"
                    break
                if self._interrupt_requested:
                    timeout_reason = "interrupted"
                    break

                if proc.returncode is not None:
                    process_exited = True

                wait_items: set[asyncio.Task[object]] = set()
                done: set[asyncio.Task[object]] = set()
                if reader_task is not None:
                    if reader_task.done():
                        done.add(reader_task)
                    else:
                        wait_items.add(reader_task)
                if wait_task is not None:
                    if wait_task.done():
                        done.add(wait_task)
                    else:
                        wait_items.add(wait_task)
                if not done and not wait_items:
                    break
                if not done:
                    wait_timeout = 0.1
                    if turn_timeout > 0:
                        wait_timeout = min(
                            wait_timeout,
                            max(0.0, turn_timeout - (now - started_at)),
                        )
                    if idle_timeout > 0:
                        wait_timeout = min(
                            wait_timeout,
                            max(0.0, idle_timeout - (now - last_activity_at)),
                        )
                    done, _pending = await asyncio.wait(
                        wait_items,
                        timeout=wait_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        continue

                if wait_task is not None and wait_task in done:
                    process_exited = True
                    wait_task = None

                if reader_task is not None and reader_task in done:
                    line = reader_task.result()
                    if not line:
                        stdout_eof = True
                        reader_task = None
                    else:
                        last_activity_at = asyncio.get_running_loop().time()
                        decoded = line.decode("utf-8", errors="replace")
                        raw_tail.append(_redact_raw_line(decoded.rstrip("\r\n")))
                        for parsed, delta_text, usage, session_id, tool_count in (
                            _parse_stream_line(decoded, assistant_text)
                        ):
                            if (
                                parsed.get("stage") == "notification"
                                and parsed.get("subtype") == "api_retry"
                            ):
                                api_retry_count += 1
                            if (
                                parsed.get("stage") == "turn_completed"
                                and parsed.get("subtype") in {"success", "done"}
                            ):
                                result_success_seen = True
                            if session_id:
                                last_session_id = session_id
                            if usage:
                                token_usage = usage
                            if tool_count:
                                tool_iterations += tool_count
                            if parsed.get("stage") == "error":
                                stream_error_kind = str(
                                    parsed.get("subtype") or parsed.get("error_kind") or "stream_error"
                                )
                                stream_error = str(
                                    parsed.get("error")
                                    or parsed.get("subtype")
                                    or delta_text
                                    or "Claude stream returned an error."
                                )
                            if delta_text:
                                if parsed.get("stage") == "error":
                                    stream_error = stream_error or delta_text
                                else:
                                    accumulated.append(delta_text)
                                    assistant_text += delta_text
                            if progress_callback is not None:
                                progress_callback(parsed)
                        reader_task = asyncio.create_task(proc.stdout.readline())

                if process_exited and stdout_eof:
                    break
        finally:
            if reader_task is not None and not reader_task.done():
                reader_task.cancel()
            if wait_task is not None and not wait_task.done():
                if proc.returncode is not None:
                    _close_subprocess_transport(proc)
                wait_task.cancel()

        if timeout_reason:
            await _kill_process(proc)
            error_text = (
                f"Claude turn timed out after {turn_timeout:g}s ({timeout_reason})"
                if timeout_reason == "hard"
                else (
                    f"Claude turn idle-timed out after {idle_timeout:g}s with no "
                    "stream activity."
                )
            )
            interrupted = timeout_reason == "interrupted"
            if interrupted:
                error_text = "Claude turn interrupted by user."
            return TurnResult(
                final_text="".join(accumulated),
                error=error_text,
                interrupted=interrupted,
                should_retire=(timeout_reason != "interrupted"),
                session_id=last_session_id,
                token_usage_last=token_usage,
                token_usage_total=token_usage,
                tool_iterations=tool_iterations,
                error_kind=timeout_reason,
                raw_output_tail=list(raw_tail),
            )

        text = "".join(accumulated)
        exit_code = proc.returncode or 0
        if stream_error:
            return TurnResult(
                final_text=text,
                error=stream_error,
                should_retire=True,
                session_id=last_session_id,
                token_usage_last=token_usage,
                token_usage_total=token_usage,
                tool_iterations=tool_iterations,
                error_kind=stream_error_kind or "stream_error",
                exit_status=exit_code,
                api_retry_count=api_retry_count,
                raw_output_tail=list(raw_tail),
            )
        if exit_code != 0:
            if result_success_seen:
                self.session_id = last_session_id
                self.thread_id = last_session_id or self.thread_id
                warning = _success_then_nonzero_warning(exit_code, api_retry_count)
                return TurnResult(
                    final_text=text,
                    warning=warning,
                    error_kind="success_result_then_exit_1",
                    exit_status=exit_code,
                    api_retry_count=api_retry_count,
                    raw_output_tail=list(raw_tail),
                    session_id=last_session_id,
                    token_usage_last=token_usage,
                    token_usage_total=token_usage,
                    tool_iterations=tool_iterations,
                )
            error_text, error_kind = _nonzero_exit_error(
                exit_code,
                api_retry_count=api_retry_count,
            )
            return TurnResult(
                final_text=text,
                error=error_text,
                should_retire=True,
                session_id=last_session_id,
                token_usage_last=token_usage,
                token_usage_total=token_usage,
                tool_iterations=tool_iterations,
                error_kind=error_kind,
                exit_status=exit_code,
                api_retry_count=api_retry_count,
                raw_output_tail=list(raw_tail),
            )
        self.session_id = last_session_id
        self.thread_id = last_session_id or self.thread_id
        return TurnResult(
            final_text=text,
            session_id=last_session_id,
            token_usage_last=token_usage,
            token_usage_total=token_usage,
            tool_iterations=tool_iterations,
            runtime="cli",
        )

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

        thread = threading.Thread(target=worker, name="claude-turn-worker", daemon=True)
        thread.start()
        while thread.is_alive():
            thread.join(timeout=0.1)
        if "error" in result_box:
            raise result_box["error"]  # type: ignore[misc]
        return result_box.get("value")  # type: ignore[return-value]

    async def _probe_capabilities(self) -> ClaudeCliCapabilities:
        if self._capabilities is not None:
            return self._capabilities
        if self._session_factory is not None:
            # Test fakes can supply a synthetic capability probe.
            self._capabilities = ClaudeCliCapabilities(
                print_prompt=True,
                output_format=True,
                stream_json_output=True,
                include_partial_messages=True,
                include_hook_events=False,
                permission_mode=True,
                model=True,
                effort=True,
                resume=True,
                session_id=True,
            )
            return self._capabilities
        self._capabilities = await probe_claude_capabilities(self._resolved_binary)
        if self._capabilities.probe_error == "binary_not_found":
            raise FileNotFoundError(
                f"Claude binary not found: {self._resolved_binary}"
            )
        if not self._capabilities.print_prompt:
            self._capabilities = ClaudeCliCapabilities.minimal()
        return self._capabilities

    def _prompt_args(
        self,
        prompt: str,
        *,
        capabilities: ClaudeCliCapabilities,
    ) -> list[str]:
        args: list[str] = []
        if self.resume_thread_id and capabilities.resume:
            args.extend(["--resume", self.resume_thread_id, "-p", prompt])
        elif self.session_id and capabilities.session_id:
            args.extend(["--session-id", self.session_id, "-p", prompt])
        else:
            args.extend(["-p", prompt])
        if capabilities.permission_mode:
            args.extend(["--permission-mode", _normalize_permission_mode(self._permission_mode)])
        if self._model and capabilities.model:
            args.extend(["--model", self._model])
        if self._effort and capabilities.effort:
            args.extend(["--effort", self._effort])
        if capabilities.output_format and capabilities.stream_json_output:
            args.extend(["--output-format", "stream-json", "--verbose"])
            if capabilities.include_partial_messages:
                args.append("--include-partial-messages")
            if capabilities.include_hook_events:
                args.append("--include-hook-events")
        return args


# ---------------------------------------------------------------------------
# Stream parsing
# ---------------------------------------------------------------------------


def _parse_stream_line(
    line: str,
    assistant_text: str,
) -> list[tuple[dict[str, Any], str, Optional[dict[str, Any]], str, int]]:
    """Parse one stream-json line into progress + delta tuples.

    Returns a list of ``(progress_event, delta_text, usage, session_id,
    tool_count)`` tuples. ``delta_text`` is non-empty only for visible text
    deltas. ``tool_count`` is 1 for each completed tool_use block.
    """
    stripped = line.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return [(
            {"stage": "notification", "method": "raw", "raw": line[:500]},
            line,
            None,
            "",
            0,
        )]
    if not isinstance(payload, dict):
        return []

    event_type = str(payload.get("type") or "")
    events: list[tuple[dict[str, Any], str, Optional[dict[str, Any]], str, int]] = []

    if event_type == "stream_event":
        events.extend(_handle_stream_event(payload))
        return events

    if event_type == "assistant":
        events.extend(_handle_assistant(payload, assistant_text))
        return events

    if event_type == "result":
        events.extend(_handle_result(payload, assistant_text))
        return events

    if event_type == "system":
        events.append((
            {
                "stage": "notification",
                "method": "system",
                "subtype": str(payload.get("subtype") or ""),
                "message": str(payload.get("message") or "")[:500],
            },
            "",
            None,
            "",
            0,
        ))
        return events

    if event_type in {"error", "api_error"}:
        text = str(payload.get("message") or payload.get("error") or payload)
        events.append((
            {"stage": "error", "error": text[:2000]},
            text,
            None,
            "",
            0,
        ))
        return events

    if event_type.startswith("hook."):
        events.append((
            {
                "stage": "notification",
                "method": event_type,
                "hook": str(payload.get("hook") or "")[:500],
            },
            "",
            None,
            "",
            0,
        ))
        return events

    events.append((
        {"stage": "notification", "method": event_type or "unknown"},
        "",
        None,
        "",
        0,
    ))
    return events


def _handle_stream_event(
    payload: dict[str, Any],
) -> list[tuple[dict[str, Any], str, Optional[dict[str, Any]], str, int]]:
    event = payload.get("event")
    if not isinstance(event, dict):
        return [(
            {"stage": "notification", "method": "stream_event"},
            "",
            None,
            "",
            0,
        )]
    inner_type = str(event.get("type") or "")
    delta = event.get("delta")
    if not isinstance(delta, dict):
        delta = {}

    if inner_type == "content_block_start":
        content_block = event.get("content_block")
        if isinstance(content_block, dict) and content_block.get("type") == "tool_use":
            tool_name = str(content_block.get("name") or "unknown")
            return [(
                {
                    "stage": "notification",
                    "method": "item/started",
                    "item": {"type": "tool_use", "name": tool_name},
                },
                "",
                None,
                "",
                0,
            )]
        return [(
            {"stage": "notification", "method": "content_block_start"},
            "",
            None,
            "",
            0,
        )]

    if delta.get("type") == "text_delta":
        text = delta.get("text")
        if isinstance(text, str) and text:
            return [(
                {"stage": "notification", "method": "text_delta"},
                text,
                None,
                "",
                0,
            )]
        return [(
            {"stage": "notification", "method": "text_delta_empty"},
            "",
            None,
            "",
            0,
        )]

    if delta.get("type") == "input_json_delta":
        return [(
            {
                "stage": "notification",
                "method": "input_json_delta",
                "partial_json": str(delta.get("partial_json", ""))[:2000],
            },
            "",
            None,
            "",
            0,
        )]

    return [(
        {"stage": "notification", "method": inner_type or "stream_event"},
        "",
        None,
        "",
        0,
    )]


def _handle_assistant(
    payload: dict[str, Any],
    assistant_text: str,
) -> list[tuple[dict[str, Any], str, Optional[dict[str, Any]], str, int]]:
    message = payload.get("message")
    if not isinstance(message, dict):
        return [(
            {"stage": "notification", "method": "assistant"},
            "",
            None,
            "",
            0,
        )]
    content = message.get("content")
    events: list[tuple[dict[str, Any], str, Optional[dict[str, Any]], str, int]] = []
    text_parts: list[str] = []
    tool_count = 0

    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
            elif item.get("type") == "tool_use":
                tool_name = str(item.get("name") or "unknown")
                events.append((
                    {
                        "stage": "notification",
                        "method": "item/completed",
                        "item": {"type": "tool_use", "name": tool_name},
                    },
                    "",
                    None,
                    "",
                    0,
                ))
                tool_count += 1

    current_text = "".join(text_parts)
    delta_text = ""
    if current_text:
        if current_text.startswith(assistant_text):
            delta_text = current_text[len(assistant_text):]
        else:
            delta_text = current_text
        events.insert(0, (
            {
                "stage": "notification",
                "method": "item/completed",
                "item": {"type": "agentMessage"},
            },
            delta_text,
            None,
            "",
            0,
        ))
    events.append((
        {"stage": "notification", "method": "assistant"},
        "",
        None,
        "",
        tool_count,
    ))
    return events


def _handle_result(
    payload: dict[str, Any],
    assistant_text: str,
) -> list[tuple[dict[str, Any], str, Optional[dict[str, Any]], str, int]]:
    events: list[tuple[dict[str, Any], str, Optional[dict[str, Any]], str, int]] = []
    subtype = str(payload.get("subtype") or "")
    session_id = str(payload.get("session_id") or "")
    usage = _extract_usage(payload)

    if subtype and subtype not in {"success", "done"}:
        error_text = str(payload.get("error") or payload.get("message") or "")
        events.append((
            {"stage": "error", "subtype": subtype, "error": error_text[:2000]},
            error_text,
            None,
            session_id,
            0,
        ))
        return events

    result_text = str(payload.get("result") or "")
    model = str(payload.get("model") or "") if isinstance(payload.get("model"), str) else ""
    usage_payload: Optional[dict[str, Any]] = None
    if usage:
        usage_payload = dict(usage)
        if model:
            usage_payload["model"] = model
        events.append((
            {
                "stage": "notification",
                "method": "thread/tokenUsage/updated",
                "usage": usage_payload,
            },
            "",
            usage_payload,
            session_id,
            0,
        ))

    if result_text and not assistant_text:
        events.append((
            {"stage": "notification", "method": "result"},
            result_text,
            None,
            session_id,
            0,
        ))

    events.append((
        {"stage": "turn_completed", "subtype": subtype, "session_id": session_id},
        "",
        None,
        session_id,
        0,
    ))
    return events


def _extract_usage(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    result: dict[str, Any] = {}
    for src_key, dst_key in [
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("cache_read_input_tokens", "cached_input_tokens"),
        ("cache_creation_input_tokens", "cached_input_tokens"),
    ]:
        val = usage.get(src_key)
        if isinstance(val, (int, float)):
            result[dst_key] = result.get(dst_key, 0) + int(val)
    if "input_tokens" in result and "output_tokens" in result:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
        result["source"] = "exact"
        return result
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_CLAUDE_ENV_DENY_LIST: tuple[str, ...] = (
    "WLCODEX_TELEGRAM_BOT_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_API_TOKEN",
    "TELEGRAM_API_ID",
    "TELEGRAM_API_HASH",
    "WLC_CHAT_ID",
    "WLCODEX_CHAT_ID",
)

_CLAUDE_ENV_DENY_SUBSTRINGS: tuple[str, ...] = (
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_API_TOKEN",
)

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


def _sanitized_env() -> dict[str, str]:
    """Return a Claude subprocess env without platform delivery secrets."""
    result = dict(os.environ)
    for key in list(result.keys()):
        if key in _CLAUDE_ENV_DENY_LIST:
            del result[key]
            continue
        for sub in _CLAUDE_ENV_DENY_SUBSTRINGS:
            if sub in key:
                del result[key]
                break
    return result


def _redact_raw_line(text: str) -> str:
    redacted = _AUTH_HEADER_RE.sub(lambda m: f"{m.group(1)}: [REDACTED]", text)
    redacted = _SECRET_ASSIGNMENT_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}[REDACTED]",
        redacted,
    )
    redacted = _BEARER_RE.sub(lambda m: f"{m.group(1)} [REDACTED]", redacted)
    redacted = _SK_SECRET_RE.sub("[REDACTED]", redacted)
    return redacted[:2000]


def _nonzero_exit_error(exit_code: int, *, api_retry_count: int) -> tuple[str, str]:
    if api_retry_count:
        return (
            "Claude API retries were exhausted before the CLI exited "
            f"with status {exit_code} ({api_retry_count} retries observed).",
            "api_retry_non_zero_exit",
        )
    return (
        f"Claude CLI exited with status {exit_code} before a successful result event.",
        "non_zero_exit",
    )


def _success_then_nonzero_warning(exit_code: int, api_retry_count: int) -> str:
    retry_note = f"; {api_retry_count} API retries observed" if api_retry_count else ""
    return (
        "Claude returned a successful result event, but the CLI process exited "
        f"with status {exit_code}{retry_note}."
    )


_PERMISSION_MODE_ALIASES: dict[str, str] = {
    "acceptEdits": "acceptEdits",
    "accept_edits": "acceptEdits",
    "auto": "auto",
    "plan": "plan",
    "default": "default",
    "dontAsk": "dontAsk",
    "dont_ask": "dontAsk",
    "bypassPermissions": "bypassPermissions",
    "bypass_permissions": "bypassPermissions",
    "yolo": "bypassPermissions",
    "full": "bypassPermissions",
}


def _normalize_permission_mode(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return "acceptEdits"
    return _PERMISSION_MODE_ALIASES.get(raw, _PERMISSION_MODE_ALIASES.get(raw.lower(), raw))


def _coerce_turn_input_text(user_input: Any) -> str:
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        parts: list[str] = []
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text"}:
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                elif item.get("type") in {"image", "image_url", "input_image"}:
                    parts.append("[image attached]")
        return "\n\n".join(part for part in parts if part).strip()
    return "" if user_input is None else str(user_input)


async def _kill_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    await proc.wait()


def _close_subprocess_transport(proc: asyncio.subprocess.Process) -> None:
    transport = getattr(proc, "_transport", None)
    if transport is None:
        return
    try:
        transport.close()
    except Exception:
        logger.debug("Failed to close subprocess transport", exc_info=True)
