"""Gateway approval bridge for Codex app-server requests."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable


class CodexApprovalError(RuntimeError):
    """Raised when Codex cannot obtain a gateway approval decision."""


class CodexApprovalBridge(AbstractContextManager["CodexApprovalBridge"]):
    """Bind Codex app-server approval requests to the active gateway chat."""

    def __init__(
        self,
        *,
        session_key: str,
        notify: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        self.session_key = session_key
        self.notify = notify
        self._token: Any = None
        self._active = False

    def __enter__(self) -> "CodexApprovalBridge":
        if not self.session_key or self.notify is None:
            return self
        from tools.approval import register_gateway_notify, set_current_session_key

        self._token = set_current_session_key(self.session_key)
        register_gateway_notify(self.session_key, self.notify)
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._active:
            return None
        from tools.approval import (
            reset_current_session_key,
            unregister_gateway_notify,
        )

        unregister_gateway_notify(self.session_key)
        if self._token is not None:
            reset_current_session_key(self._token)
        self._active = False
        return None

    def callback(
        self,
        command: str,
        description: str,
        *,
        allow_permanent: bool = False,
    ) -> str:
        if not self.session_key or self.notify is None:
            raise CodexApprovalError("Codex 审批通道不可用")

        from tools.approval import request_gateway_approval

        decision = request_gateway_approval(
            {
                "command": command,
                "description": description,
                "pattern_key": "codex_app_server_approval",
                "pattern_keys": ["codex_app_server_approval"],
                "allow_permanent": bool(allow_permanent),
            },
            surface="codex_app_server",
        )
        if decision.get("notify_failed"):
            raise CodexApprovalError("Codex 审批通道不可用：通知发送失败")
        if decision.get("reason") == "no_notify_callback":
            raise CodexApprovalError("Codex 审批通道不可用：没有通知回调")
        if decision.get("reason") == "no_session":
            raise CodexApprovalError("Codex 审批通道不可用：没有当前会话")
        if not decision.get("resolved"):
            raise CodexApprovalError("Codex 审批已超时")
        choice = str(decision.get("choice") or "deny")
        if choice == "deny":
            raise CodexApprovalError("Codex 审批已拒绝")
        return choice
