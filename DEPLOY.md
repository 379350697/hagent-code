# Hermes Codex Control Plane Deployment

This repository is intentionally small. It contains only the Codex control
plane code and its focused tests. It does not vendor Hermes gateway files,
Discord adapters, Telegram adapters, or secrets.

## Layout

```text
gateway/control_planes/codex/   # platform-neutral Codex command core
tests/gateway/                  # focused architecture/service tests
```

## Install Into A Hermes Checkout

Set `HERMES_AGENT` to the root of a Hermes checkout, for example:

```bash
export HERMES_AGENT=/home/wl/.hermes/hermes-agent
```

Copy the control plane package:

```bash
mkdir -p "$HERMES_AGENT/gateway/control_planes"
cp -R gateway/control_planes/codex "$HERMES_AGENT/gateway/control_planes/"
cp gateway/control_planes/__init__.py "$HERMES_AGENT/gateway/control_planes/__init__.py"
```

Remove the retired WebSocket tool wrapper if it exists in that checkout:

```bash
rm -f "$HERMES_AGENT/tools/codex_app_server.py"
rm -f "$HERMES_AGENT/tests/tools/test_codex_app_server.py"
```

Do not remove `agent/transports/codex_app_server.py` or
`agent/transports/codex_app_server_session.py`; those are the stdio runtime
used by the control plane.

## Minimal Hermes Integration

Hermes should keep platform code in its original locations. The only required
gateway integration is for the existing `/codex` command handler to delegate to
the control plane:

```python
from gateway.control_planes.codex import CommandRequest, get_codex_command_service

source = event.source if event else None
platform = source.platform.value if source and source.platform else "unknown"
request = CommandRequest(
    platform=platform,
    chat_id=str(getattr(source, "chat_id", "") or ""),
    user_id=str(getattr(source, "user_id", "") or ""),
    thread_id=str(getattr(source, "thread_id", "") or ""),
    text=event.get_command_args().strip() if event else "",
    workspace=getattr(self, "working_directory", None) or os.getcwd(),
)
result = await get_codex_command_service().handle(request)
return result.text
```

Recommended Discord adapter behavior:

- Defer the slash interaction first.
- If defer fails, log `Discord interaction failed` and do not call the Codex
  service.
- Keep unauthorized Discord logging local to Discord. Do not notify Telegram or
  Slack from inside the Discord adapter.
- Register Discord `/codex` as native subcommands and dispatch them as text
  commands such as `/codex new <task>`, `/codex plan <task>`, and
  `/codex continue <task>`.

Telegram should dispatch the same text commands through the existing Hermes
slash-command path. It should not import Discord adapter code.

## Workspace Selection

The control plane supports per-platform/chat/thread workspace selection:

```text
/codex workspace
/codex workspace current
/codex workspace set <number-or-path>
/codex workspace clear
```

`/codex workspace` lists discovered local git repositories. Discovery checks
`HERMES_CODEX_WORKSPACE_ROOTS` first, then `CODEX_WORKSPACE_ROOTS`. Both use
the platform path separator, for example:

```bash
export HERMES_CODEX_WORKSPACE_ROOTS="/home/wl/projects:/home/wl/.hermes"
```

If neither variable is set, discovery scans the Hermes workspace, `HERMES_HOME`,
and the user home with common heavy directories skipped.

## Runtime Config

Use platform-specific environment variables in Hermes:

```bash
DISCORD_BOT_TOKEN=...
DISCORD_ALLOWED_USERS=...
DISCORD_HOME=...

TELEGRAM_BOT_TOKEN=...
TELEGRAM_ALLOWED_USERS=...
TELEGRAM_HOME=...
```

Do not commit secrets.

## Validate

From the Hermes checkout after copying the package:

```bash
python3 -m py_compile \
  gateway/control_planes/__init__.py \
  gateway/control_planes/codex/*.py

venv/bin/python -m pytest \
  tests/gateway/test_codex_command_service.py \
  tests/gateway/test_codex_control_architecture.py
```

For a full Hermes integration, also run the platform tests that cover Discord
slash authorization and startup/reconnect isolation.

Restart the Hermes gateway after deployment. A still-running old process can
keep serving the retired `/codex` tool path until it is restarted.
