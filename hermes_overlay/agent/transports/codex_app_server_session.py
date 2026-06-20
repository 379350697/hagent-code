"""Session adapter for codex app-server runtime.

Owns one Codex thread per Hermes session. Drives `turn/start`, consumes
streaming notifications via CodexEventProjector, handles server-initiated
approval requests (apply_patch, exec command), translates cancellation,
and returns a clean turn result that AIAgent.run_conversation() can splice
into its `messages` list.

Lifecycle:
    session = CodexAppServerSession(cwd="/home/x/proj")
    session.ensure_started()                              # spawns + handshake + thread/start
    result = session.run_turn(user_input="hello")         # blocks until turn/completed
    # result.final_text          → assistant text returned to caller
    # result.projected_messages  → list of {role, content, ...} for messages list
    # result.tool_iterations     → how many tool-shaped items completed (skill nudge counter)
    # result.interrupted         → True if Ctrl+C / interrupt_requested fired mid-turn
    session.close()                                       # tears down subprocess

Threading model: the adapter is single-threaded from the caller's perspective.
The underlying CodexAppServerClient owns its own reader threads but exposes
blocking-with-timeout queues that this adapter polls in a loop, so the run_turn
call is synchronous and behaves like AIAgent's existing chat_completions loop.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from agent.codex_responses_adapter import _format_responses_error
from agent.redact import redact_sensitive_text
from agent.transports.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
)
from agent.transports.codex_event_projector import CodexEventProjector

logger = logging.getLogger(__name__)


# How many tailing stderr lines from the codex subprocess to attach to a
# user-facing error when we don't have a more specific classification (OAuth,
# wedge watchdog, etc.). Small enough to keep error messages legible, large
# enough to surface a config/provider/auth diagnostic.
_STDERR_TAIL_LINES = 12


def _read_codex_model_default() -> str:
    """Read model from ~/.codex/config.toml, falling back to 'gpt-5.5'."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    try:
        with open(os.path.expanduser("~/.codex/config.toml"), "rb") as f:
            return str(tomllib.load(f).get("model", "gpt-5.5"))
    except Exception:
        return "gpt-5.5"


# Permission profile mapping mirrors the docstring in PR proposal:
# Hermes' tools.terminal.security_mode → Codex's permissions profile id.
# Defaults if config is missing → workspace-write (matches Codex's own default).
_HERMES_TO_CODEX_PERMISSION_PROFILE = {
    "auto": "workspace-write",
    "approval-required": "read-only-with-approval",
    "unrestricted": "full-access",
    # Backstop alias used by some skills/tests.
    "yolo": "full-access",
}


@dataclass
class TurnResult:
    """Result of one user→assistant→tool turn through the codex app-server."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None  # Set if turn ended in a non-recoverable error
    turn_id: Optional[str] = None
    thread_id: Optional[str] = None
    token_usage_last: Optional[dict[str, Any]] = None
    token_usage_total: Optional[dict[str, Any]] = None
    model_context_window: Optional[int] = None
    # Hint to the caller that the underlying codex subprocess is likely
    # wedged (turn-level timeout fired, post-tool watchdog tripped, or
    # token-refresh failure killed the child). The caller should retire
    # the session so the next turn respawns codex from scratch instead
    # of riding a CPU-spinning or auth-broken process. Mirrors openclaw
    # beta.8's "retire timed-out app-server clients" fix.
    should_retire: bool = False


# Markers we accept as terminal even when codex never emits turn/completed.
# Some codex versions stream `<turn_aborted>` as raw text in agentMessage
# items when an interrupt or upstream error tears the turn down before the
# normal completion path fires. Mirrors openclaw beta.8 fix.
_TURN_ABORTED_MARKERS = ("<turn_aborted>", "<turn_aborted/>")


def _coerce_turn_input_text(user_input: Any) -> str:
    """Collapse Hermes/OpenAI rich content into app-server text input.

    The current `turn/start` path sends text items only. TUI image attachment
    can hand us OpenAI-style content parts, so keep the text/path hints and
    replace opaque image payloads with a small marker instead of putting a
    Python list into the `text` field.
    """
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        parts: list[str] = []
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                if item is not None:
                    parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item_type in {"image", "image_url", "input_image"}:
                parts.append("[image attached]")
        text = "\n\n".join(p for p in parts if p).strip()
        return text or "What do you see in this image?"
    return "" if user_input is None else str(user_input)


# Substrings in codex stderr / JSON-RPC error messages that signal the
# subprocess died because its OAuth credentials are no longer valid.
# Kept conservative: we only redirect users to `codex login` when we're
# reasonably sure that's the actual failure, otherwise we surface the
# original error verbatim. Mirrors openclaw beta.8's auth-refresh
# classification.
_OAUTH_REFRESH_FAILURE_HINTS = (
    "invalid_grant",
    "invalid grant",
    "refresh token",
    "refresh_token",
    "token refresh",
    "token_refresh",
    "token has expired",
    "expired_token",
    "expired token",
    "not authenticated",
    "unauthenticated",
    "unauthorized",
    "401 unauthorized",
    "re-authenticate",
    "reauthenticate",
    "please log in",
    "please login",
    "auth profile",
    "no auth profile",
    "oauth",
)


def _classify_oauth_failure(*parts: str) -> Optional[str]:
    """Return a user-friendly re-auth hint if any of the provided strings
    look like a codex OAuth/token-refresh failure; otherwise None.

    Used for both `turn/start` JSON-RPC errors and post-mortem stderr
    inspection when the subprocess exits unexpectedly. Conservative on
    purpose — we only redirect users to `codex login` when the signal
    is strong, so unrelated runtime failures still surface verbatim.
    """
    haystack = " ".join(p for p in parts if p).lower()
    if not haystack:
        return None
    for needle in _OAUTH_REFRESH_FAILURE_HINTS:
        if needle in haystack:
            return (
                "Codex authentication failed — your ChatGPT/Codex login "
                "looks expired or invalid. Run `codex login` to refresh, "
                "then retry. (Fall back to default runtime with "
                "`/codex-runtime auto` if the issue persists.)"
            )
    return None


@dataclass
class _ServerRequestRouting:
    """Default policies for codex-side approval requests when no interactive
    callback is wired in. These are only used by tests + cron / non-interactive
    contexts; the live CLI path passes an approval_callback that defers to
    tools.approval.prompt_dangerous_approval()."""

    auto_approve_exec: bool = False
    auto_approve_apply_patch: bool = False


class CodexAppServerSession:
    """One Codex thread per Hermes session, lifetime owned by AIAgent.

    Not thread-safe — one caller drives it at a time, matching how AIAgent's
    run_conversation() loop is structured today. The codex client itself can
    handle interleaved reads/writes via its own threads, but the adapter's
    state (projector, thread_id, turn counter) is owned by the caller thread.
    """

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        codex_bin: str = "codex",
        codex_home: Optional[str] = None,
        permission_profile: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        config_overrides: Optional[list[str]] = None,
        resume_thread_id: str = "",
        on_event: Optional[Callable[[dict], None]] = None,
        request_routing: Optional[_ServerRequestRouting] = None,
        client_factory: Optional[Callable[..., CodexAppServerClient]] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._codex_bin = codex_bin
        self._codex_home = codex_home
        self._permission_profile = (
            permission_profile or _HERMES_TO_CODEX_PERMISSION_PROFILE.get(
                os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto"),
                "workspace-write",
            )
        )
        self._approval_callback = approval_callback
        self._config_overrides = list(config_overrides or [])
        self._resume_thread_id = str(resume_thread_id or "")
        self._on_event = on_event  # Display hook (kawaii spinner ticks etc.)
        self._routing = request_routing or _ServerRequestRouting()
        self._client_factory = client_factory or CodexAppServerClient

        self._client: Optional[CodexAppServerClient] = None
        self._thread_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        # Pending file-change items, keyed by item id. Populated on
        # item/started for fileChange items; consumed by the approval
        # bridge when codex sends item/fileChange/requestApproval. The
        # approval params don't carry the changeset, so we cache here
        # to surface a real summary in the approval prompt (quirk #4).
        self._pending_file_changes: dict[str, str] = {}
        self._last_approval_error: Optional[str] = None
        self._closed = False

    # ---------- lifecycle ----------

    def ensure_started(self) -> str:
        """Spawn the subprocess, do the initialize handshake, and start/resume a
        thread. Returns the codex thread id. Idempotent — repeated calls
        return the same thread id."""
        if self._thread_id is not None:
            return self._thread_id
        if self._client is None:
            self._client = self._client_factory(
                codex_bin=self._codex_bin,
                codex_home=self._codex_home,
                extra_args=self._config_overrides,
            )
        self._client.initialize(
            client_name="hermes",
            client_title="Hermes Agent",
            client_version=_get_hermes_version(),
        )
        if self._resume_thread_id:
            self._thread_id = self._resume_thread()
            return self._thread_id

        self._thread_id = self._start_thread()
        return self._thread_id

    def _start_thread(self) -> str:
        # Permission selection is intentionally NOT sent on thread/start.
        # Two reasons (live-tested against codex 0.130.0):
        #   1. `thread/start.permissions` is gated behind the experimentalApi
        #      capability on this codex version — we'd have to opt in during
        #      initialize and accept the unstable surface.
        #   2. Even with experimentalApi declared and the correct shape
        #      (`{"type": "profile", "id": "..."}`, not `{"profileId": ...}`),
        #      codex requires a matching `[permissions]` table in
        #      ~/.codex/config.toml or it fails the request with
        #      'default_permissions requires a [permissions] table'.
        # Letting codex pick its default (`:read-only` unless the user has
        # configured otherwise in their codex config.toml) is the standard
        # codex CLI workflow and avoids fighting codex's own validation.
        # Users who want a write-capable profile configure it in their
        # ~/.codex/config.toml the same way they would for any codex usage.
        params: dict[str, Any] = {"cwd": self._cwd}
        # Codex ≥ 0.141 requires model on thread/start.
        params.setdefault("model", _read_codex_model_default())
        assert self._client is not None
        result = self._client.request("thread/start", params, timeout=15)
        # Cross-fill thread.id/sessionId — different codex versions have
        # serialized this under either key. Mirrors openclaw beta.8's
        # tolerance fix so future codex drops/renames don't KeyError us
        # at handshake time.
        thread_id = _extract_thread_id(result)
        if not thread_id:
            raise CodexAppServerError(
                code=-32603,
                message=(
                    "codex thread/start returned no thread id "
                    f"(payload keys: {sorted(result.keys())})"
                ),
            )
        logger.info(
            "codex app-server thread started: id=%s profile=%s cwd=%s",
            thread_id[:8],
            self._permission_profile,
            self._cwd,
        )
        return thread_id

    def _resume_thread(self) -> str:
        assert self._client is not None
        params: dict[str, Any] = {
            "threadId": self._resume_thread_id,
            "cwd": self._cwd,
            "model": _read_codex_model_default(),
        }
        result = self._client.request("thread/resume", params, timeout=60)
        thread_id = _extract_thread_id(result) or self._resume_thread_id
        if not thread_id:
            raise CodexAppServerError(
                code=-32603,
                message=(
                    "codex thread/resume returned no thread id "
                    f"(payload keys: {sorted(result.keys())})"
                ),
            )
        logger.info(
            "codex app-server thread resumed: id=%s profile=%s cwd=%s",
            thread_id[:8],
            self._permission_profile,
            self._cwd,
        )
        return thread_id

    def set_approval_callback(self, callback: Optional[Callable[..., str]]) -> None:
        self._approval_callback = callback

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            self._client = None
        self._thread_id = None

    def __enter__(self) -> "CodexAppServerSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def request_interrupt(self) -> None:
        """Idempotent: signal the active turn loop to issue turn/interrupt
        and unwind. Called by AIAgent's _interrupt_requested path."""
        self._interrupt_event.set()

    # ---------- diagnostics ----------

    def _format_error_with_stderr(
        self,
        prefix: str,
        exc: Any = "",
        *,
        tail_lines: int = _STDERR_TAIL_LINES,
    ) -> str:
        """Build a user-facing error string for codex failures.

        Appends the last few lines of codex's stderr buffer when available,
        passed through agent.redact with force=True so secrets in provider
        error responses (auth headers, query-string tokens, sk-* keys) never
        leak into chat output or trajectories. The codex CLI's own error
        text ('Internal error', 'turn/start failed: ...') is otherwise
        opaque and forces users to re-run with verbose flags to diagnose
        config / provider / auth-bridge problems.

        Use this for the generic / catch-all branches. Specific
        classifications (OAuth via _classify_oauth_failure, post-tool wedge
        watchdog) already produce a clean hint and should be used instead.
        """
        exc_str = str(exc) if exc != "" and exc is not None else ""
        base = f"{prefix}: {exc_str}" if exc_str else prefix
        if self._client is None:
            return base
        try:
            tail = self._client.stderr_tail(tail_lines)
        except Exception:  # pragma: no cover - diagnostic best-effort
            return base
        if not tail:
            return base
        joined = "\n".join(line.rstrip() for line in tail if line)
        if not joined.strip():
            return base
        redacted = redact_sensitive_text(joined, force=True)
        return f"{base}\ncodex stderr (last {len(tail)} lines):\n{redacted}"

    # ---------- per-turn ----------

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: float = 600.0,
        notification_poll_timeout: float = 0.25,
        post_tool_quiet_timeout: float = 90.0,
        active_tool_timeout: float = 3600.0,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        progress_interval: float = 60.0,
    ) -> TurnResult:
        """Send a user message and block until turn/completed, while
        forwarding server-initiated approval requests and projecting items
        into Hermes' messages shape.

        post_tool_quiet_timeout: if codex emits a tool completion and then
        goes quiet for this many seconds without emitting another item or
        `turn/completed`, fast-fail and mark the session for retirement.
        Mirrors openclaw beta.8's post-tool completion watchdog (#81697)
        so a wedged codex doesn't burn the full turn deadline.
        """
        # Pre-create the result so startup failures (codex subprocess can't
        # spawn, initialize handshake rejects, thread/start blows up) surface
        # the same way per-turn failures do — with a TurnResult.error string
        # the caller can render — instead of bubbling raw codex exceptions
        # up to AIAgent.run_conversation.
        def emit_progress(stage: str, **payload: Any) -> None:
            if progress_callback is None:
                return
            data = {
                "stage": stage,
                "cwd": self._cwd,
                "thread_id": self._thread_id or "",
                "turn_id": result.turn_id or "",
            }
            data.update(payload)
            try:
                progress_callback(data)
            except Exception:  # pragma: no cover - progress callbacks are best-effort
                logger.debug("progress_callback raised", exc_info=True)

        result = TurnResult()
        try:
            self.ensure_started()
        except (CodexAppServerError, TimeoutError) as exc:
            result.error = self._format_error_with_stderr(
                "codex app-server startup failed", exc
            )
            # Subprocess almost certainly unhealthy — retire so the next
            # turn re-spawns cleanly.
            result.should_retire = True
            return result
        assert self._client is not None and self._thread_id is not None
        result.thread_id = self._thread_id
        emit_progress("session_ready")

        self._interrupt_event.clear()
        self._last_approval_error = None
        projector = CodexEventProjector()

        user_input_text = _coerce_turn_input_text(user_input)

        # Send turn/start with the user input. Text-only for now (codex
        # supports rich content but Hermes' text path is the common case).
        turn_started_wall_at = time.time()
        try:
            ts = self._client.request(
                "turn/start",
                {
                    "threadId": self._thread_id,
                    "input": [{"type": "text", "text": user_input_text}],
                },
                timeout=10,
            )
        except CodexAppServerError as exc:
            # Classify auth/refresh failures so the user gets a clear
            # `codex login` pointer instead of a raw RPC error string.
            stderr_blob = "\n".join(self._client.stderr_tail(40))
            hint = _classify_oauth_failure(exc.message, stderr_blob)
            if hint is not None:
                result.error = hint
                # Subprocess is fine on a JSON-RPC level here, but the
                # token store is broken — retire so the next turn does a
                # clean handshake (and the user has a chance to re-auth
                # via `codex login` between turns).
                result.should_retire = True
            else:
                result.error = self._format_error_with_stderr(
                    "turn/start failed", exc
                )
            return result
        except TimeoutError as exc:
            # turn/start hanging is a strong signal the subprocess is wedged.
            stderr_blob = "\n".join(self._client.stderr_tail(40))
            hint = _classify_oauth_failure(stderr_blob)
            result.error = hint or self._format_error_with_stderr(
                "turn/start timed out", exc
            )
            result.should_retire = True
            return result

        result.turn_id = (ts.get("turn") or {}).get("id")
        last_activity_at = time.monotonic()
        last_progress_at = 0.0
        turn_complete = False
        # Post-tool watchdog state. last_tool_completion_at is set whenever
        # a tool-shaped item completes; if no further notification arrives
        # within post_tool_quiet_timeout and the turn hasn't completed, we
        # fast-fail and retire the session.
        last_tool_completion_at: Optional[float] = None
        active_tools: dict[str, tuple[float, str]] = {}

        emit_progress("turn_started")

        while not turn_complete:
            now = time.monotonic()
            if active_tools:
                oldest_started_at, oldest_label = min(
                    active_tools.values(),
                    key=lambda item: item[0],
                )
                if (now - oldest_started_at) > active_tool_timeout:
                    if self._recover_completed_turn(
                        result,
                        since_epoch=turn_started_wall_at,
                        progress_callback=emit_progress,
                    ):
                        turn_complete = True
                        break
                    self._issue_interrupt(result.turn_id)
                    result.interrupted = True
                    diagnostic = _recover_native_recent_diagnostic(
                        self._thread_id or result.thread_id or "",
                        codex_home=self._codex_home,
                        since_epoch=turn_started_wall_at,
                    )
                    result.error = (
                        f"Codex 命令运行超过 {active_tool_timeout:.0f} 秒仍未完成："
                        f"{oldest_label}"
                    )
                    if diagnostic:
                        result.error += f"\n原生 Codex 最近活动：{diagnostic}"
                    result.should_retire = True
                    break
            elif (now - last_activity_at) > turn_timeout:
                break
            if self._interrupt_event.is_set():
                self._issue_interrupt(result.turn_id)
                result.interrupted = True
                break

            # Detect a dead subprocess between iterations. If codex exited
            # (e.g. crashed, segfaulted, or its auth refresh thread killed
            # the process), we won't get any more notifications — bail out
            # rather than waiting for the full turn deadline.
            if not self._client.is_alive():
                stderr_blob = "\n".join(self._client.stderr_tail(60))
                hint = _classify_oauth_failure(stderr_blob)
                if hint is not None:
                    result.error = hint
                else:
                    result.error = self._format_error_with_stderr(
                        "codex app-server subprocess exited unexpectedly",
                        tail_lines=20,
                    )
                result.should_retire = True
                break

            # Post-tool watchdog: if a tool completion was the most recent
            # signal and codex has been silent past the quiet timeout, give
            # up on this turn instead of waiting for the outer deadline.
            if (
                not active_tools
                and
                last_tool_completion_at is not None
                and (time.monotonic() - last_tool_completion_at)
                    > post_tool_quiet_timeout
            ):
                if self._recover_completed_turn(
                    result,
                    since_epoch=turn_started_wall_at,
                    progress_callback=emit_progress,
                ):
                    turn_complete = True
                    break
                self._issue_interrupt(result.turn_id)
                result.interrupted = True
                diagnostic = _recover_native_recent_diagnostic(
                    self._thread_id or result.thread_id or "",
                    codex_home=self._codex_home,
                    since_epoch=turn_started_wall_at,
                )
                result.error = (
                    f"Codex app-server 在工具步骤后 "
                    f"{post_tool_quiet_timeout:.0f} 秒没有新事件；"
                    f"已回收本轮运行时。"
                )
                if diagnostic:
                    result.error += f"\n原生 Codex 最近活动：{diagnostic}"
                result.should_retire = True
                break

            # Drain any server-initiated requests (approvals) before
            # reading notifications, so the codex side isn't blocked.
            sreq = self._client.take_server_request(timeout=0)
            if sreq is not None:
                last_activity_at = time.monotonic()
                emit_progress(
                    "approval_requested",
                    method=str(sreq.get("method") or ""),
                    request=sreq,
                )
                emit_progress(
                    "server_request",
                    method=str(sreq.get("method") or ""),
                    request=sreq,
                )
                # Drain any pending notifications first so per-turn state
                # (e.g. _pending_file_changes for fileChange approvals) is
                # up to date when we make the approval decision. Bounded
                # to avoid starving the server-request response.
                for _ in range(8):
                    pending = self._client.take_notification(timeout=0)
                    if pending is None:
                        break
                    last_activity_at = time.monotonic()
                    _track_active_tool_notification(pending, active_tools)
                    if pending.get("method") == "turn/completed":
                        active_tools.clear()
                        turn_complete = True
                    if _is_meaningful_post_tool_activity(pending):
                        last_tool_completion_at = None
                    emit_progress(
                        "notification",
                        method=str(pending.get("method") or ""),
                        notification=pending,
                    )
                    _apply_token_usage_notification(result, pending)
                    self._track_pending_file_change(pending)
                    proj = projector.project(pending)
                    if proj.messages:
                        result.projected_messages.extend(proj.messages)
                    if proj.is_tool_iteration:
                        result.tool_iterations += 1
                        last_tool_completion_at = time.monotonic()
                        emit_progress(
                            "tool_completed",
                            method=str(pending.get("method") or ""),
                            tool_iterations=result.tool_iterations,
                        )
                    if proj.final_text is not None:
                        result.final_text = proj.final_text
                        if _has_turn_aborted_marker(proj.final_text):
                            turn_complete = True
                            result.interrupted = True
                            result.error = (
                                result.error
                                or "codex reported turn_aborted"
                            )
                self._handle_server_request(sreq)
                # Activity counts as live signal — reset the post-tool
                # quiet timer so an approval round-trip doesn't trip it.
                last_tool_completion_at = None
                continue

            note = self._client.take_notification(
                timeout=notification_poll_timeout
            )
            if note is None:
                if progress_callback is not None and progress_interval > 0:
                    now = time.monotonic()
                    if (now - last_progress_at) > progress_interval:
                        last_progress_at = now
                        emit_progress(
                            "waiting",
                            idle_seconds=max(0.0, now - last_activity_at),
                            timeout_seconds=turn_timeout,
                            **_active_tool_progress_payload(active_tools, now),
                        )
                continue

            last_activity_at = time.monotonic()
            method = note.get("method", "")
            _track_active_tool_notification(note, active_tools)
            if _is_meaningful_post_tool_activity(note):
                last_tool_completion_at = None
            emit_progress(
                "notification",
                method=method,
                notification=note,
            )
            if self._on_event is not None:
                try:
                    self._on_event(note)
                except Exception:  # pragma: no cover - display callback
                    logger.debug("on_event callback raised", exc_info=True)

            _apply_token_usage_notification(result, note)

            # Track in-progress fileChange items so the approval bridge
            # can surface a real change summary when codex requests
            # approval (the approval params themselves don't carry the
            # changeset). Quirk #4 fix.
            self._track_pending_file_change(note)

            # Project into messages
            projection = projector.project(note)
            if projection.messages:
                result.projected_messages.extend(projection.messages)
            if projection.is_tool_iteration:
                result.tool_iterations += 1
                # Arm/refresh the post-tool quiet watchdog whenever a
                # tool-shaped item completes.
                last_tool_completion_at = time.monotonic()
                emit_progress(
                    "tool_completed",
                    method=method,
                    tool_iterations=result.tool_iterations,
                )
            else:
                # Any non-tool projected activity (assistant message,
                # status update, etc.) means codex is still producing
                # output — clear the quiet timer so we don't fast-fail.
                if projection.messages or projection.final_text is not None:
                    last_tool_completion_at = None
            if projection.final_text is not None:
                # Codex can emit multiple agentMessage items in one turn
                # (e.g. partial then final). Take the last one as canonical.
                result.final_text = projection.final_text
                # Some codex builds tear a turn down by emitting a
                # `<turn_aborted>` marker in the agent message text and
                # never sending turn/completed. Treat the marker itself
                # as terminal so we don't burn the full deadline.
                if _has_turn_aborted_marker(projection.final_text):
                    turn_complete = True
                    result.interrupted = True
                    result.error = (
                        result.error or "codex reported turn_aborted"
                    )

            if method == "turn/completed":
                emit_progress("turn_completed", method=method)
                turn_complete = True
                turn_status = (
                    (note.get("params") or {}).get("turn") or {}
                ).get("status")
                if turn_status and turn_status not in {"completed", "interrupted"}:
                    err_obj = (
                        (note.get("params") or {}).get("turn") or {}
                    ).get("error")
                    if err_obj:
                        err_msg = _format_responses_error(err_obj, str(turn_status))
                        # If the turn failed for an auth/refresh reason,
                        # rewrite the error into a re-auth hint AND mark
                        # the session for retirement.
                        stderr_blob = "\n".join(
                            self._client.stderr_tail(40)
                        )
                        hint = _classify_oauth_failure(err_msg, stderr_blob)
                        if hint is not None:
                            result.error = hint
                            result.should_retire = True
                        else:
                            result.error = self._format_error_with_stderr(
                                f"turn ended status={turn_status}", err_msg
                            )

        if not turn_complete and not result.interrupted:
            if self._recover_completed_turn(
                result,
                since_epoch=turn_started_wall_at,
                progress_callback=emit_progress,
            ):
                return result
            # Hit the inactivity deadline. Issue interrupt to stop wasted
            # compute, and tell the caller to retire the session — a turn
            # that never produced another app-server signal is a strong sign
            # codex is wedged in a way the next turn shouldn't inherit.
            self._issue_interrupt(result.turn_id)
            result.interrupted = True
            if not result.error:
                result.error = self._format_error_with_stderr(
                    f"turn timed out after {turn_timeout}s without app-server activity"
                )
            result.should_retire = True
            emit_progress("turn_timed_out", timeout_seconds=turn_timeout)

        if self._last_approval_error and not result.error:
            result.error = self._last_approval_error

        return result

    # ---------- internals ----------

    def _recover_completed_turn(
        self,
        result: TurnResult,
        *,
        since_epoch: float,
        progress_callback: Callable[..., None],
    ) -> bool:
        """Trust Codex's native task ledger when app-server completion is lost.

        Hermes is only an observer of the app-server stream. If the stream
        wedges after Codex has already written `task_complete` to its own
        session JSONL, reporting "Codex failed" is wrong. In that case return
        the native final answer, retire this app-server process, and let the
        next turn resume the same thread cleanly.
        """
        recovered = _recover_native_task_complete(
            self._thread_id or result.thread_id or "",
            codex_home=self._codex_home,
            since_epoch=since_epoch,
        )
        if not recovered:
            return False
        result.final_text = recovered
        result.error = None
        result.interrupted = False
        result.should_retire = True
        _append_recovered_assistant_message(result, recovered)
        progress_callback(
            "turn_completion_recovered",
            source="codex_native_session",
        )
        return True

    def _issue_interrupt(self, turn_id: Optional[str]) -> None:
        if self._client is None or self._thread_id is None or turn_id is None:
            return
        try:
            self._client.request(
                "turn/interrupt",
                {"threadId": self._thread_id, "turnId": turn_id},
                timeout=5,
            )
        except CodexAppServerError as exc:
            # "no active turn to interrupt" is fine — already done.
            logger.debug("turn/interrupt non-fatal: %s", exc)
        except TimeoutError:
            logger.warning("turn/interrupt timed out")

    def _handle_server_request(self, req: dict) -> None:
        """Translate a codex server request (approval) into Hermes' approval
        flow, then send the response.

        Method names verified live against codex 0.130.0 (Apr 2026):
          item/commandExecution/requestApproval — exec approvals
          item/fileChange/requestApproval       — apply_patch approvals
          item/permissions/requestApproval      — permissions changes
                                                  (we decline; user controls
                                                  permission profile in
                                                  ~/.codex/config.toml).
        """
        if self._client is None:
            return
        method = req.get("method", "")
        rid = req.get("id")
        params = req.get("params") or {}

        if method == "item/commandExecution/requestApproval":
            decision = self._decide_exec_approval(params)
            self._client.respond(rid, {"decision": decision})
        elif method == "item/fileChange/requestApproval":
            decision = self._decide_apply_patch_approval(params)
            self._client.respond(rid, {"decision": decision})
        elif method == "item/permissions/requestApproval":
            # Codex sometimes asks to escalate permissions mid-turn. We
            # always decline — the user already chose their permission
            # profile in ~/.codex/config.toml and surprise escalations
            # shouldn't be silently accepted.
            self._client.respond(rid, {"decision": "decline"})
        elif method == "mcpServer/elicitation/request":
            # Codex's MCP layer asks the user for structured input on
            # behalf of an MCP server (e.g. tool-call confirmation,
            # OAuth, form data). For our own hermes-tools callback we
            # auto-accept — the user already approved Hermes' tools
            # by enabling the runtime, and we never expose anything
            # codex's built-in shell can't already do. For other MCP
            # servers we decline so the user explicitly opts in via
            # codex's own auth flow.
            server_name = params.get("serverName") or ""
            if server_name == "hermes-tools":
                self._client.respond(
                    rid,
                    {"action": "accept", "content": None, "_meta": None},
                )
            else:
                self._client.respond(
                    rid,
                    {"action": "decline", "content": None, "_meta": None},
                )
        else:
            # Unknown server request — codex can extend this surface. Reject
            # cleanly so codex doesn't hang waiting for us.
            logger.warning("Unknown codex server request: %s", method)
            self._client.respond_error(
                rid, code=-32601, message=f"Unsupported method: {method}"
            )

    def _decide_exec_approval(self, params: dict) -> str:
        if self._routing.auto_approve_exec:
            return "accept"
        command = params.get("command") or ""
        # Codex's CommandExecutionRequestApprovalParams has cwd as Optional —
        # fall back to the session's cwd when codex doesn't include it so the
        # approval prompt is never empty (quirk #10 fix).
        cwd = params.get("cwd") or self._cwd or "<unknown>"
        reason = params.get("reason")
        description = f"Codex 请求在 {cwd} 执行命令"
        if reason:
            description += f" — {reason}"
        if self._approval_callback is not None:
            try:
                choice = self._approval_callback(
                    command, description, allow_permanent=False
                )
                return _approval_choice_to_codex_decision(choice)
            except Exception as exc:
                self._last_approval_error = str(exc) or "Codex 审批被拒绝"
                logger.exception("approval_callback raised on exec request")
                return "decline"
        self._last_approval_error = "Codex 审批通道不可用"
        return "decline"  # fail-closed when no callback wired

    def _decide_apply_patch_approval(self, params: dict) -> str:
        if self._routing.auto_approve_apply_patch:
            return "accept"
        if self._approval_callback is not None:
            # FileChangeRequestApprovalParams gives us reason + grantRoot.
            # The actual changeset lives on the corresponding fileChange
            # item which the projector has already cached for us — look it
            # up by item_id so the user sees what's actually changing.
            reason = params.get("reason")
            grant_root = params.get("grantRoot")
            item_id = params.get("itemId") or ""
            change_summary = self._lookup_pending_file_change(item_id)
            description_parts = []
            if reason:
                description_parts.append(reason)
            if change_summary:
                description_parts.append(change_summary)
            if grant_root:
                description_parts.append(f"需要写入权限：{grant_root}")
            description = (
                "; ".join(description_parts)
                if description_parts
                else "Codex 请求应用文件修改"
            )
            command_label = (
                f"apply_patch: {change_summary}" if change_summary
                else f"apply_patch: {reason}" if reason
                else "apply_patch"
            )
            try:
                choice = self._approval_callback(
                    command_label,
                    description,
                    allow_permanent=False,
                )
                return _approval_choice_to_codex_decision(choice)
            except Exception as exc:
                self._last_approval_error = str(exc) or "Codex 审批被拒绝"
                logger.exception("approval_callback raised on apply_patch")
                return "decline"
        self._last_approval_error = "Codex 审批通道不可用"
        return "decline"

    def _track_pending_file_change(self, note: dict) -> None:
        """Maintain self._pending_file_changes from item/started + item/completed
        notifications. Lets the apply_patch approval prompt show what's
        actually changing — codex's approval params don't carry the data."""
        method = note.get("method", "")
        params = note.get("params") or {}
        item = params.get("item") or {}
        if item.get("type") != "fileChange":
            return
        item_id = item.get("id") or ""
        if not item_id:
            return
        if method == "item/started":
            changes = item.get("changes") or []
            if not changes:
                self._pending_file_changes[item_id] = "1 change pending"
                return
            kinds: dict[str, int] = {}
            paths: list[str] = []
            for ch in changes:
                if not isinstance(ch, dict):
                    continue
                kind = (ch.get("kind") or {}).get("type") or "update"
                kinds[kind] = kinds.get(kind, 0) + 1
                p = ch.get("path") or ""
                if p:
                    paths.append(p)
            counts = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
            preview = ", ".join(paths[:3])
            if len(paths) > 3:
                preview += f", +{len(paths) - 3} more"
            self._pending_file_changes[item_id] = (
                f"{counts}: {preview}" if preview else counts
            )
        elif method == "item/completed":
            self._pending_file_changes.pop(item_id, None)

    def _lookup_pending_file_change(self, item_id: str) -> Optional[str]:
        """Look up an in-progress fileChange item by id and summarize its
        changes for the approval prompt. Returns None when we don't have
        the item cached (e.g. approval arrived before item/started, or
        fileChange item content not tracked yet)."""
        if not item_id:
            return None
        cached = self._pending_file_changes.get(item_id)
        if not cached:
            return None
        return cached


def _apply_token_usage_notification(result: TurnResult, note: dict) -> None:
    """Capture Codex app-server token usage updates for caller accounting.

    Codex does not put token usage on turn/completed. It emits a separate
    thread/tokenUsage/updated notification containing cumulative totals and
    the latest turn breakdown.
    """
    if not isinstance(note, dict) or note.get("method") != "thread/tokenUsage/updated":
        return
    params = note.get("params") or {}
    token_usage = params.get("tokenUsage") or {}
    if not isinstance(token_usage, dict):
        return
    last = token_usage.get("last")
    total = token_usage.get("total")
    if isinstance(last, dict):
        result.token_usage_last = dict(last)
    if isinstance(total, dict):
        result.token_usage_total = dict(total)
    window = token_usage.get("modelContextWindow")
    if isinstance(window, int) and window > 0:
        result.model_context_window = window


def _approval_choice_to_codex_decision(choice: str) -> str:
    """Map Hermes approval choices onto codex's CommandExecutionApprovalDecision
    / FileChangeApprovalDecision wire values.

    Hermes returns 'once', 'session', 'always', or 'deny'.
    Codex expects 'accept', 'acceptForSession', 'decline', or 'cancel'
    (verified against codex-rs/app-server-protocol/src/protocol/v2/item.rs
    on codex 0.130.0).
    """
    if choice in {"once",}:
        return "accept"
    if choice in {"session", "always"}:
        return "acceptForSession"
    return "decline"


def _is_meaningful_post_tool_activity(notification: dict[str, Any]) -> bool:
    """Return True when a notification proves Codex is still producing work."""
    method = str((notification or {}).get("method") or "")
    return method not in {
        "",
        "thread/tokenUsage/updated",
        "account/rateLimits/updated",
    }


def _track_active_tool_notification(
    notification: dict[str, Any],
    active_tools: dict[str, tuple[float, str]],
) -> None:
    method = str((notification or {}).get("method") or "")
    params = notification.get("params") if isinstance(notification, dict) else {}
    params = params if isinstance(params, dict) else {}
    item = params.get("item") if isinstance(params.get("item"), dict) else {}
    item_type = str(item.get("type") or "")
    item_id = str(item.get("id") or params.get("itemId") or "")
    if item_type not in {"commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall"}:
        return
    if not item_id:
        item_id = f"{item_type}:{len(active_tools) + 1}"
    if method == "item/started":
        active_tools[item_id] = (time.monotonic(), _tool_label(item))
    elif method == "item/completed":
        active_tools.pop(item_id, None)


def _active_tool_progress_payload(
    active_tools: dict[str, tuple[float, str]],
    now: float,
) -> dict[str, Any]:
    if not active_tools:
        return {}
    started_at, label = min(active_tools.values(), key=lambda item: item[0])
    return {
        "active_tool_label": label,
        "active_tool_elapsed_seconds": max(0.0, now - started_at),
    }


def _tool_label(item: dict[str, Any]) -> str:
    command = str(item.get("command") or "")
    if command:
        return redact_sensitive_text(" ".join(command.split()), force=True)[:160]
    name = str(item.get("name") or item.get("toolName") or item.get("type") or "工具步骤")
    return name[:160]


def _append_recovered_assistant_message(result: TurnResult, text: str) -> None:
    if not text:
        return
    for message in result.projected_messages:
        if (
            message.get("role") == "assistant"
            and str(message.get("content") or "") == text
        ):
            return
    result.projected_messages.append({"role": "assistant", "content": text})


def _recover_native_task_complete(
    thread_id: str,
    *,
    codex_home: Optional[str],
    since_epoch: float,
) -> str:
    """Return the latest native Codex `task_complete` text for this turn.

    Codex's own CLI/app-server writes a JSONL rollout under
    `$CODEX_HOME/sessions`. That file is the source of truth for whether the
    model produced a final answer. We only accept entries written after the
    current turn started so an old completed turn is never mistaken for the
    current one.
    """
    if not thread_id:
        return ""
    home = codex_home or os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    pattern = os.path.join(home, "sessions", "**", f"*{thread_id}.jsonl")
    candidates = sorted(glob.glob(pattern, recursive=True))
    if not candidates:
        return ""

    best_text = ""
    best_timestamp = 0.0
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    timestamp = _parse_jsonl_timestamp(record.get("timestamp"))
                    if timestamp and timestamp < since_epoch - 5.0:
                        continue
                    text = _native_task_complete_text(record)
                    if not text:
                        continue
                    if timestamp >= best_timestamp:
                        best_timestamp = timestamp
                        best_text = text
        except OSError:
            logger.debug("failed to read codex native session %s", path, exc_info=True)
    return redact_sensitive_text(best_text.strip(), force=True) if best_text else ""


def _recover_native_recent_diagnostic(
    thread_id: str,
    *,
    codex_home: Optional[str],
    since_epoch: float,
) -> str:
    """Return the newest useful native Codex activity after this turn began.

    This is deliberately diagnostic-only: unlike `task_complete` recovery it
    never changes the turn status. It gives remote users a concrete last
    native event when the app-server observer goes quiet.
    """
    if not thread_id:
        return ""
    home = codex_home or os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    pattern = os.path.join(home, "sessions", "**", f"*{thread_id}.jsonl")
    best_text = ""
    best_timestamp = 0.0
    best_order = -1
    order = 0
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    order += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    timestamp = _parse_jsonl_timestamp(record.get("timestamp"))
                    if timestamp and timestamp < since_epoch - 5.0:
                        continue
                    text = _native_diagnostic_text(record)
                    if not text:
                        continue
                    if timestamp > best_timestamp or (
                        timestamp == best_timestamp and order >= best_order
                    ):
                        best_timestamp = timestamp
                        best_order = order
                        best_text = text
        except OSError:
            logger.debug("failed to read codex native session %s", path, exc_info=True)
    if not best_text:
        return ""
    return redact_sensitive_text(_compact_diagnostic(best_text), force=True)


def _native_diagnostic_text(record: dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return ""
    record_type = record.get("type")
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    if record_type == "response_item":
        item = payload.get("item") if isinstance(payload.get("item"), dict) else payload
        item_type = str(item.get("type") or "")
        if item_type == "function_call_output":
            output = item.get("output")
            return _native_output_diagnostic(output)
        if item_type in {"function_call", "local_shell_call"}:
            name = str(item.get("name") or item.get("call_type") or item_type)
            args = item.get("arguments") or item.get("input") or ""
            preview = _compact_diagnostic(str(args), limit=180) if args else ""
            return f"原生工具调用 {name}: {preview}" if preview else f"原生工具调用 {name}"
    if record_type == "event_msg":
        event_type = str(payload.get("type") or "")
        if event_type == "turn_aborted":
            return "原生 turn_aborted"
        if event_type in {"error", "exec_error", "task_failed"}:
            message = payload.get("message") or payload.get("error") or payload
            return f"原生事件 {event_type}: {message}"
    return ""


def _native_output_diagnostic(output: Any) -> str:
    if output is None:
        return ""
    text = str(output)
    lines = [_compact_diagnostic(line, limit=240) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    error_markers = (
        "traceback",
        "error",
        "failed",
        "exception",
        "syntaxerror",
        "permission denied",
        "operation not permitted",
        "timed out",
        "command not found",
        "bwrap",
    )
    for line in reversed(lines):
        lowered = line.lower()
        if any(marker in lowered for marker in error_markers):
            return f"原生工具输出：{line}"
    return f"原生工具输出：{lines[-1]}"


def _compact_diagnostic(text: str, *, limit: int = 320) -> str:
    value = " ".join(str(text or "").split())
    if len(value) > limit:
        return value[: limit - 3] + "..."
    return value


def _native_task_complete_text(record: dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return ""
    if record.get("type") == "event_msg":
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        if payload.get("type") == "task_complete":
            return str(payload.get("last_agent_message") or "").strip()
    return ""


def _parse_jsonl_timestamp(value: Any) -> float:
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _extract_thread_id(payload: dict[str, Any]) -> str:
    """Return a Codex thread id from start/resume response variants."""
    if not isinstance(payload, dict):
        return ""
    thread_obj = payload.get("thread") if isinstance(payload.get("thread"), dict) else {}
    value = (
        thread_obj.get("id")
        or thread_obj.get("sessionId")
        or payload.get("sessionId")
        or payload.get("threadId")
    )
    return str(value or "")


def _has_turn_aborted_marker(text: str) -> bool:
    """Return True if `text` contains any of the raw markers codex uses
    to signal a turn was aborted without emitting `turn/completed`.

    Codex emits `<turn_aborted>` (and sometimes `<turn_aborted/>`) as raw
    text inside agentMessage items when an interrupt or upstream error
    tears the turn down before the normal completion path fires. Mirrors
    openclaw beta.8's terminal-marker fix so we don't burn the full turn
    deadline waiting for a turn/completed that never comes.
    """
    if not text:
        return False
    for marker in _TURN_ABORTED_MARKERS:
        if marker in text:
            return True
    return False


def _get_hermes_version() -> str:
    """Best-effort Hermes version string for codex's userAgent line."""
    try:
        from importlib.metadata import version

        return version("hermes-agent")
    except Exception:  # pragma: no cover
        return "0.0.0"
