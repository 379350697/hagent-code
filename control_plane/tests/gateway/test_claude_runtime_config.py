import json

from gateway.control_planes.claude.runtime_config import (
    claude_runtime,
    claude_runtime_fallback,
    claude_sdk_profile_name,
    resolve_claude_sdk_profile,
    safe_claude_sdk_profile_diagnostics,
)


def test_claude_runtime_defaults_to_sdk_with_cli_fallback(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_CLAUDE_RUNTIME", raising=False)
    monkeypatch.delenv("HERMES_CLAUDE_RUNTIME_FALLBACK", raising=False)
    assert claude_runtime({}) == "agent_sdk"
    assert claude_runtime_fallback({}) == "cli"


def test_opencodego_profile_reads_claude_settings_without_leaking_secret(
    tmp_path,
    monkeypatch,
) -> None:
    claude_home = tmp_path / ".claude"
    claude_home.mkdir()
    (claude_home / "settings.json").write_text(
        json.dumps({
            "env": {
                "ANTHROPIC_API_KEY": "sk-super-secret",
                "ANTHROPIC_AUTH_TOKEN": "sk-super-secret",
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:15721",
                "ANTHROPIC_MODEL": "deepseek-v4-pro",
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr("os.path.expanduser", lambda value: str(claude_home) if value == "~/.claude" else value.replace("~", str(tmp_path)))

    profile = resolve_claude_sdk_profile({"sdk_profile": "opencodego"})
    diagnostics = safe_claude_sdk_profile_diagnostics(profile)

    assert profile["api_key"] == "sk-super-secret"
    assert diagnostics["api_key_available"] is True
    assert "sk-super-secret" not in json.dumps(diagnostics)
    assert diagnostics["base_url"] == "http://127.0.0.1:15721"


def test_deepseek_profile_defaults(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_CLAUDE_SDK_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_CLAUDE_SDK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    profile = resolve_claude_sdk_profile({"sdk_profile": "deepseek"})
    diagnostics = safe_claude_sdk_profile_diagnostics(profile)
    assert profile["base_url"] == "https://api.deepseek.com/anthropic"
    assert profile["api_key_env"] == "DEEPSEEK_API_KEY"
    assert diagnostics["api_key_available"] is True


def test_anthropic_profile_uses_anthropic_api_key_env(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_CLAUDE_SDK_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_CLAUDE_SDK_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    profile = resolve_claude_sdk_profile({"sdk_profile": "anthropic"})
    diagnostics = safe_claude_sdk_profile_diagnostics(profile)
    assert profile["base_url"] == ""
    assert profile["api_key_env"] == "ANTHROPIC_API_KEY"
    assert diagnostics["api_key_available"] is True


def test_sdk_profile_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_CLAUDE_SDK_PROFILE", "deepseek")
    monkeypatch.setenv("HERMES_CLAUDE_SDK_BASE_URL", "https://example.test/anthropic")
    monkeypatch.setenv("HERMES_CLAUDE_SDK_API_KEY_ENV", "CUSTOM_KEY")
    monkeypatch.setenv("HERMES_CLAUDE_SDK_MODEL", "custom-model")
    assert claude_sdk_profile_name({}) == "deepseek"
    profile = resolve_claude_sdk_profile({})
    assert profile["base_url"] == "https://example.test/anthropic"
    assert profile["api_key_env"] == "CUSTOM_KEY"
    assert profile["model"] == "custom-model"


def test_sdk_profile_passes_explicit_claude_binary(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_CLAUDE_BINARY", "/opt/claude")
    profile = resolve_claude_sdk_profile({})
    assert profile["cli_path"] == "/opt/claude"
