from gateway.control_planes.claude.session_pool import ClaudeSessionPool


class FakeRuntimeSession:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.cwd = kwargs["cwd"]
        self.thread_id = kwargs.get("resume_thread_id", "")
        self.session_id = self.thread_id
        self.closed = False

    def ensure_started(self):
        return self.thread_id

    def run_turn(self, user_input, **options):
        raise AssertionError("not used")

    def request_interrupt(self):
        pass

    def close(self):
        self.closed = True

    def set_approval_callback(self, callback):
        pass


def test_claude_session_pool_retires_when_runtime_changes() -> None:
    pool = ClaudeSessionPool(session_factory=FakeRuntimeSession)
    first = pool.get(
        "discord:1:main",
        "/repo",
        new_session=True,
        runtime="agent_sdk",
        runtime_fallback="cli",
        sdk_profile="opencodego",
    )
    second = pool.get(
        "discord:1:main",
        "/repo",
        new_session=False,
        runtime="cli",
        runtime_fallback="",
        sdk_profile="",
        resume_thread_id="session-1",
    )

    assert first is not None
    assert second is not None
    assert first is not second
    assert first.session.closed
    assert second.runtime == "cli"


def test_claude_session_pool_retires_when_sdk_profile_changes() -> None:
    pool = ClaudeSessionPool(session_factory=FakeRuntimeSession)
    first = pool.get(
        "discord:1:main",
        "/repo",
        new_session=True,
        runtime="agent_sdk",
        runtime_fallback="cli",
        sdk_profile="opencodego",
    )
    second = pool.get(
        "discord:1:main",
        "/repo",
        new_session=False,
        runtime="agent_sdk",
        runtime_fallback="cli",
        sdk_profile="deepseek",
        resume_thread_id="session-1",
    )

    assert first is not None
    assert second is not None
    assert first is not second
    assert first.session.closed
    assert second.sdk_profile == "deepseek"
