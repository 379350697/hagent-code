#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  verify-hermes-overlay.sh --hermes-agent PATH

Verifies that the hagent-code overlay is installed into a Hermes checkout.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
hermes_agent=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hermes-agent)
      hermes_agent="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$hermes_agent" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -d "$hermes_agent" ]]; then
  echo "Hermes checkout not found: $hermes_agent" >&2
  exit 1
fi

python_bin="$hermes_agent/venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

required=(
  "gateway/control_planes/codex/service.py"
  "gateway/control_planes/codex/repair.py"
  "gateway/control_planes/claude/service.py"
  "gateway/control_planes/claude/narrator.py"
  "agent/transports/codex_app_server_session.py"
  "agent/transports/claude_cli_session.py"
  "gateway/slash_commands.py"
  "plugins/platforms/discord/adapter.py"
  "tools/approval.py"
)

for rel in "${required[@]}"; do
  if [[ ! -f "$hermes_agent/$rel" ]]; then
    echo "Missing required file: $rel" >&2
    exit 1
  fi
done

mapfile -t py_files < <(
  {
    find "$hermes_agent/gateway/control_planes/codex" -type f -name '*.py'
    find "$hermes_agent/gateway/control_planes/claude" -type f -name '*.py'
    printf '%s\n' \
      "$hermes_agent/agent/transports/codex_app_server_session.py" \
      "$hermes_agent/agent/transports/claude_cli_session.py" \
      "$hermes_agent/gateway/slash_commands.py" \
      "$hermes_agent/gateway/run.py" \
      "$hermes_agent/gateway/platforms/api_server.py" \
      "$hermes_agent/hermes_cli/commands.py" \
      "$hermes_agent/plugins/platforms/discord/adapter.py" \
      "$hermes_agent/tools/approval.py"
  } | sort
)

"$python_bin" -m py_compile "${py_files[@]}"

if [[ -x "$hermes_agent/venv/bin/python" ]]; then
  (
    cd "$hermes_agent"
    "$python_bin" -m pytest \
      tests/gateway/test_codex_command_service.py \
      tests/gateway/test_codex_control_architecture.py \
      tests/gateway/test_claude_command_service.py \
      tests/gateway/test_claude_control_architecture.py \
      tests/agent/transports/test_codex_app_server_session.py \
      tests/gateway/test_discord_slash_commands.py \
      -q
  )
else
  echo "No Hermes venv found; skipped pytest."
fi

echo "Overlay verification passed."
