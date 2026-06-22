"""Factory and fallback wrapper for Claude runtimes."""

from __future__ import annotations

from typing import Any, Callable, Optional

from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession
from agent.transports.claude_cli_session import ClaudeCliSession
from agent.transports.claude_runtime import ClaudeRuntimeSession, TurnResult


def create_claude_runtime_session(
    *,
    cwd: str,
    config_overrides: Optional[list[str]] = None,
    resume_thread_id: str = "",
    permission_mode: str = "acceptEdits",
    model: str = "",
    effort: str = "",
    runtime: str = "agent_sdk",
    runtime_fallback: str = "cli",
    sdk_profile_config: Optional[dict[str, Any]] = None,
    sdk_runner: Optional[Any] = None,
    cli_factory: Optional[Callable[..., ClaudeRuntimeSession]] = None,
    **kwargs: Any,
) -> ClaudeRuntimeSession:
    runtime = _normalize_runtime(runtime)
    runtime_fallback = _normalize_runtime(runtime_fallback)
    cli_cls = cli_factory or ClaudeCliSession
    if runtime == "cli":
        return cli_cls(
            cwd=cwd,
            config_overrides=config_overrides,
            resume_thread_id=resume_thread_id,
            permission_mode=permission_mode,
            model=model,
            effort=effort,
            **kwargs,
        )
    if runtime == "agent_sdk":
        primary = ClaudeAgentSdkSession(
            cwd=cwd,
            config_overrides=config_overrides,
            resume_thread_id=resume_thread_id,
            permission_mode=permission_mode,
            model=model,
            effort=effort,
            sdk_profile_config=sdk_profile_config,
            runner=sdk_runner,
            **kwargs,
        )
        fallback = None
        if runtime_fallback == "cli":
            fallback = cli_cls(
                cwd=cwd,
                config_overrides=config_overrides,
                resume_thread_id=resume_thread_id,
                permission_mode=permission_mode,
                model=model,
                effort=effort,
                **kwargs,
            )
        return ClaudeFallbackSession(
            primary=primary,
            fallback=fallback,
            primary_runtime="agent_sdk",
            fallback_runtime=runtime_fallback,
        )
    raise ValueError(f"Unsupported Claude runtime: {runtime}")


class ClaudeFallbackSession:
    """Primary runtime with an optional fallback used before side effects start."""

    def __init__(
        self,
        *,
        primary: ClaudeRuntimeSession,
        fallback: ClaudeRuntimeSession | None,
        primary_runtime: str,
        fallback_runtime: str,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._active = primary
        self._primary_runtime = primary_runtime
        self._fallback_runtime = fallback_runtime
        self._fallback_reason = ""

    @property
    def cwd(self) -> str:
        return self._active.cwd

    @property
    def thread_id(self) -> str:
        return self._active.thread_id

    @thread_id.setter
    def thread_id(self, value: str) -> None:
        self._active.thread_id = value

    @property
    def session_id(self) -> str:
        return self._active.session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._active.session_id = value

    def ensure_started(self) -> str:
        try:
            return self._active.ensure_started()
        except Exception as exc:
            if self._active is self._primary and self._fallback is not None:
                self._fallback_reason = str(exc)
                self._active = self._fallback
                return self._active.ensure_started()
            raise

    def set_approval_callback(self, callback):
        self._active.set_approval_callback(callback)
        if self._fallback is not None and self._fallback is not self._active:
            self._fallback.set_approval_callback(callback)

    def close(self) -> None:
        self._primary.close()
        if self._fallback is not None:
            self._fallback.close()

    def request_interrupt(self) -> None:
        self._active.request_interrupt()

    def run_turn(self, user_input: Any, **options: Any) -> TurnResult:
        result = self._active.run_turn(user_input, **options)
        if (
            self._active is self._primary
            and self._fallback is not None
            and result.error
            and not result.started
        ):
            self._fallback_reason = result.error
            self._active = self._fallback
            result = self._active.run_turn(user_input, **options)
        if self._active is self._fallback and self._fallback_reason:
            result.fallback_runtime = _runtime_name(self._fallback_runtime)
            result.fallback_reason = self._fallback_reason
            result.warning = result.warning or (
                f"Claude SDK unavailable; fell back to {result.fallback_runtime}."
            )
        return result


def _normalize_runtime(value: str) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"sdk", "agent_sdk"}:
        return "agent_sdk"
    if raw in {"", "none", "off"}:
        return ""
    if raw in {"cli", "claude_cli", "claude_code"}:
        return "cli"
    return raw


def _runtime_name(value: str) -> str:
    return _normalize_runtime(value) or "none"
