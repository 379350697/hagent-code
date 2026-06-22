---
name: claude
description: "Delegate coding to Anthropic Claude Code CLI (features, PRs)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Claude, Anthropic, Code-Review, Refactoring]
    related_skills: [codex, hermes-agent]
---

# Claude Code

Delegate coding tasks to [Claude Code](https://docs.anthropic.com/en/docs/claude-code) via the Hermes gateway. Claude Code is Anthropic's autonomous coding agent CLI.

## When to use

- Building features
- Refactoring
- PR reviews
- Batch issue fixing
- Long-running implementations that benefit from Claude's reasoning

Requires the `claude` CLI and a git repository.

## Prerequisites

- Claude Code installed: `npm install -g @anthropic-ai/claude-code`
  - Or set `HERMES_CLAUDE_BINARY` to the VS Code extension's native binary
    (e.g. `~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude`)
- Anthropic auth configured: either `ANTHROPIC_API_KEY` or Claude OAuth
  credentials from `claude login`
- **Must run inside a git repository** — Claude Code refuses to run outside one
- Use `--permission-mode acceptEdits` (default) for non-interactive runs

## Hermes Session Commands

Use explicit session commands when the user wants Hermes to delegate through a
Claude session. Treat these as user intent for Hermes/Claude delegation; do not
require the user to know hidden phrases.

| Command | Action |
|---------|--------|
| `claude continue <task>` | Resume the current Claude session for this project/thread |
| `claude new <task>` | Start a fresh Claude session |
| `claude status` | Show the current Claude workspace, model, permission mode, and session id if known |
| `claude sessions` | List recent/known Claude sessions when Hermes has that context available |
| `claude select <n>` | Pick a session by index from `/claude sessions` |
| `claude resume <n> <task>` | Resume session #n with a new task |
| `claude doctor` | Run health checks on the Claude control plane |
| `claude workspace [list\|set <path>\|current\|clear]` | Manage the workspace Claude runs in |
| `claude diff` | Show a git digest of the current workspace |
| `claude stop` | Interrupt the running Claude turn |
| `claude plan <task>` | Ask Claude to plan only (no file changes) |
| `claude permissions <default\|approve-for-me\|read-only\|full-access>` | Switch permission mode |

Default behavior:

1. Same project and same conversation thread: continue the current Claude session.
2. New unrelated task: start a new Claude session.
3. Ambiguous request: show the current session context, then default to continue.

Backward-compatible aliases:

| Alias | Treat as |
|-------|----------|
| `claude 接着 <task>` | `claude continue <task>` |
| `claude 新开 <task>` | `claude new <task>` |

## How Hermes drives Claude

Unlike Codex (which uses a long-lived app-server subprocess), each Claude turn
spawns a fresh `claude -p` subprocess with `--output-format stream-json
--verbose`. Continuation across turns is done via `--resume <session_id>` which
Claude resolves from its local session store under `~/.claude/projects/`.

The platform-neutral control plane lives in
`gateway/control_planes/claude/` and the transport adapter lives in
`agent/transports/claude_cli_session.py`. Runtime events are persisted to
`~/.hermes/claude-control-plane/events.sqlite3` and task records to
`~/.hermes/claude-control-plane/tasks.json`.

## Permission Modes

| Mode | Effect |
|------|--------|
| `acceptEdits` (default) | Auto-accept file edits within the workspace |
| `auto` | Let Claude decide which operations are safe to auto-execute |
| `plan` | Planning mode — Claude proposes but does not modify files |
| `default` | Claude's default confirmation strategy |
| `dontAsk` | Do not prompt for any operation |
| `bypassPermissions` | Skip all permission checks (high risk) |

Use `/claude permissions approve-for-me` to align with the desktop app's
"auto-approve" setting (`auto` mode).

## Background Mode (Long Tasks)

Claude turns are synchronous from Hermes' perspective — the gateway blocks
until the subprocess exits. For very long tasks:

1. Use `/claude new <task>` to start the turn.
2. Hermes will stream progress notifications to the chat as Claude emits
   `stream_event` / `assistant` / `result` lines.
3. Use `/claude status` to inspect the current state and recent events.
4. Use `/claude stop` to interrupt a stuck turn.

## Key Flags (applied automatically by Hermes)

| Flag | Effect |
|------|--------|
| `-p, --print` | Non-interactive execution, exits when done |
| `--output-format stream-json --verbose` | Structured streaming output |
| `--include-partial-messages` | Stream partial text deltas |
| `--resume <session_id>` | Continue an existing conversation |
| `--session-id <uuid>` | Pin a specific session id for a new conversation |
| `--permission-mode <mode>` | Set the permission mode |
| `--model <model>` | Override the model |
| `--effort <level>` | Set reasoning effort (low/medium/high/xhigh/max) |

## Hermes Gateway Caveat

When invoking Claude Code from a Hermes gateway/service context, ensure the
subprocess environment does not leak platform delivery secrets. The transport
adapter sanitizes the environment (strips `TELEGRAM_BOT_TOKEN` and similar)
before spawning `claude`.

If `bwrap`/sandbox errors appear (unlikely for Claude, but possible in
containerized deployments), use `/claude permissions full-access` to run with
`bypassPermissions` mode and rely on process boundaries as the safety layer.

## Rules

1. **Git repo required** — Claude won't run outside a git directory. Use `mktemp -d && git init` for scratch
2. **Use `/claude new` for new tasks** — Starts a fresh session with no prior context
3. **Use `/claude continue` for follow-ups** — Resumes the current session with `--resume`
4. **Be patient with long tasks** — Monitor with `/claude status`, not by spamming
5. **Review diffs before committing** — Use `/claude diff` to see what Claude changed
6. **Permission modes are sticky** — `/claude permissions` affects subsequent turns until changed
