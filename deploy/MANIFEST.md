# HAgent Code Overlay Manifest

This manifest defines the files owned by this repository.

## Control Plane

```text
control_plane/gateway/control_planes/__init__.py
control_plane/gateway/control_planes/codex/
control_plane/gateway/control_planes/claude/
control_plane/tests/gateway/test_codex_command_service.py
control_plane/tests/gateway/test_codex_control_architecture.py
control_plane/tests/gateway/test_claude_command_service.py
control_plane/tests/gateway/test_claude_control_architecture.py
```

## Hermes Overlay

```text
hermes_overlay/agent/transports/codex_app_server_session.py
hermes_overlay/agent/transports/claude_agent_sdk_session.py
hermes_overlay/agent/transports/claude_cli_session.py
hermes_overlay/agent/transports/claude_runtime.py
hermes_overlay/agent/transports/claude_runtime_factory.py
hermes_overlay/gateway/slash_commands.py
hermes_overlay/gateway/run.py
hermes_overlay/gateway/platforms/api_server.py
hermes_overlay/hermes_cli/commands.py
hermes_overlay/plugins/platforms/discord/adapter.py
hermes_overlay/skills/autonomous-ai-agents/codex/SKILL.md
hermes_overlay/skills/autonomous-ai-agents/claude/SKILL.md
hermes_overlay/tools/approval.py
hermes_overlay/tests/agent/transports/test_codex_app_server_session.py
hermes_overlay/tests/gateway/test_approve_deny_commands.py
hermes_overlay/tests/gateway/test_discord_component_auth.py
hermes_overlay/tests/gateway/test_discord_slash_auth.py
hermes_overlay/tests/gateway/test_discord_slash_commands.py
```

## Excluded

```text
.env
.git/
__pycache__/
.pytest_cache/
tools/registry.py
tests/tools/test_registry_walk.py
```

`tools/registry.py` is intentionally excluded because it is not part of the
Hermes Codex control path.
