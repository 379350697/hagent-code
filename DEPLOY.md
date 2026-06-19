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
    approval_session_key=self._session_key_for_source(source) if source else "",
    approval_chat_id=str(getattr(source, "chat_id", "") or ""),
    approval_thread_metadata=self._thread_metadata_for_source(source) if source else {},
    approval_notify=codex_approval_notify_callback,
)
result = await get_codex_command_service().handle(request)
return result.text
```

`codex_approval_notify_callback` must register with Hermes' existing approval
queue by sending the approval prompt to the originating adapter. Prefer
`adapter.send_exec_approval(...)` when available; otherwise send a text fallback
that tells the user to reply with `/approve` or `/deny`.

Recommended Discord adapter behavior:

- Defer the slash interaction first.
- If defer fails, log `Discord interaction failed` and do not call the Codex
  service.
- Keep unauthorized Discord logging local to Discord. Do not notify Telegram or
  Slack from inside the Discord adapter.
- Register Discord `/codex` as native subcommands and dispatch them as text
  commands such as `/codex new <task>`, `/codex plan <task>`, and
  `/codex continue <task>`.
- For workspace selection, prefer a nested native command group:
  `/codex workspace list`, `/codex workspace current`,
  `/codex workspace set <repo>`, and `/codex workspace clear`. The `repo`
  parameter should use Discord autocomplete backed by
  `discover_git_workspaces(workspace_scan_roots(...))`; send the selected
  repository as `/codex workspace set <number-or-path>`.

Telegram should dispatch the same text commands through the existing Hermes
slash-command path. It should not import Discord adapter code.

Required Hermes transport integration:

- `CodexAppServerSession` must accept `config_overrides: list[str]` and pass
  them to `CodexAppServerClient(extra_args=...)`.
- It must accept `resume_thread_id: str`. When set, `ensure_started()` must
  initialize the app-server and call `thread/resume` with the stored
  `threadId` instead of `thread/start`.
- It must expose `set_approval_callback(callback)` so each `/codex` turn can
  bind a fresh gateway approval context.
- When Codex approval is unavailable, denied, or times out, surface that as a
  turn error instead of silently declining and letting the model guess.

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

Hidden git repositories such as `.hermes` and `.nvm` are skipped in discovery
so user-facing workspace choices look like project repositories. They can still
be selected manually by absolute path when needed.

## Session Semantics

`completed` means the last Codex turn completed; it does not mean the Codex
thread/session is closed.

The control plane stores the current selected session per platform/chat/thread
in:

```text
$HERMES_HOME/codex-control-plane/selected_sessions.json
```

`/codex new <task>` always starts a new Codex thread and selects it for the
current chat. Workspace selection affects new sessions only.

`/codex continue <task>` targets the selected session. If no selection exists,
it falls back to the newest recoverable session in the current chat. If the
Hermes gateway was restarted and no app-server process is live, the control
plane recreates the app-server and resumes the stored Codex `threadId`.

History commands:

```text
/codex sessions
/codex sessions workspace <query>
/codex sessions all
/codex select <number-or-task-id-or-thread-id>
/codex resume <number-or-task-id-or-thread-id> <task>
```

`/codex sessions` shows recent sessions for the current platform/chat/thread,
including workspace and last-turn status. `select` changes the current chat's
selected session without starting a turn. `resume` selects a historical session
and immediately starts a new turn in that thread. Selectors may be the displayed
number, a `task_id` prefix, or a Codex `thread_id` prefix; ambiguous prefixes
return candidates and do not start Codex.

`/codex sessions all` is diagnostic-only. Hermes should pass
`CommandRequest.is_admin=True` only for slash users who are admins under the
current platform/scope policy. Without that flag, the command is rejected unless
`HERMES_CODEX_DIAGNOSTICS=1` is explicitly set for the gateway process.

Discord should expose `/codex select` and `/codex resume` as native subcommands
with autocomplete backed by the control plane's session list. Telegram can use
the same text commands.

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

Codex app-server runtime config lives under `codex_app_server`:

```yaml
codex_app_server:
  sandbox: workspace-write   # workspace-write | read-only | danger-full-access
  approval_policy: on-request # on-request | never
```

The control plane maps `sandbox` to Codex `-c sandbox_mode="..."`. `never`
should only be used by the explicit `/codex permissions danger` path.

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
