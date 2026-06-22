from types import SimpleNamespace

from agent.transports.claude_runtime import TurnResult
from agent.transports.claude_runtime_factory import create_claude_runtime_session


class FakeCliSession:
    def __init__(self, **kwargs):
        self.cwd = kwargs["cwd"]
        self.thread_id = kwargs.get("resume_thread_id", "") or "cli-session"
        self.session_id = self.thread_id
        self.closed = False
        self.turns = []

    def ensure_started(self):
        return self.thread_id

    def run_turn(self, user_input, **options):
        self.turns.append(user_input)
        return TurnResult(
            final_text="cli ok",
            session_id=self.session_id,
            runtime="cli",
            started=True,
        )

    def request_interrupt(self):
        pass

    def close(self):
        self.closed = True

    def set_approval_callback(self, callback):
        pass


def test_runtime_factory_falls_back_to_cli_when_sdk_preflight_fails() -> None:
    session = create_claude_runtime_session(
        cwd="/tmp",
        runtime="agent_sdk",
        runtime_fallback="cli",
        sdk_profile_config={"name": "opencodego", "api_key_env": "NO_SUCH_KEY"},
        cli_factory=FakeCliSession,
    )

    assert session.ensure_started() == "cli-session"
    result = session.run_turn("hello")

    assert result.final_text == "cli ok"
    assert result.runtime == "cli"
    assert result.fallback_runtime == "cli"
    assert result.fallback_reason
