#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  install-hermes-overlay.sh --hermes-agent PATH --dry-run
  install-hermes-overlay.sh --hermes-agent PATH --apply

Copies hagent-code control_plane/ and hermes_overlay/ into a Hermes checkout.
--apply creates timestamped backups under:
  $HERMES_AGENT/.hagent-code-backups/YYYYMMDD-HHMMSS/
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
hermes_agent=""
mode=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hermes-agent)
      hermes_agent="${2:-}"
      shift 2
      ;;
    --dry-run)
      mode="dry-run"
      shift
      ;;
    --apply)
      mode="apply"
      shift
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

if [[ -z "$hermes_agent" || -z "$mode" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -d "$hermes_agent" ]]; then
  echo "Hermes checkout not found: $hermes_agent" >&2
  exit 1
fi

if [[ ! -f "$hermes_agent/hermes_cli/commands.py" ]]; then
  echo "Not a Hermes agent checkout: $hermes_agent" >&2
  exit 1
fi

backup_dir=""
if [[ "$mode" == "apply" ]]; then
  backup_dir="$hermes_agent/.hagent-code-backups/$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$backup_dir"
fi

copy_one() {
  local src="$1"
  local rel="$2"
  local dst="$hermes_agent/$rel"

  if [[ "$mode" == "dry-run" ]]; then
    if [[ -e "$dst" ]]; then
      printf 'would update  %s\n' "$rel"
    else
      printf 'would create  %s\n' "$rel"
    fi
    return
  fi

  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" ]]; then
    mkdir -p "$backup_dir/$(dirname "$rel")"
    cp -p "$dst" "$backup_dir/$rel"
  fi
  cp -p "$src" "$dst"
  printf 'installed     %s\n' "$rel"
}

install_tree() {
  local root="$1"
  local prefix="$2"
  while IFS= read -r -d '' src; do
    local rel="${src#"$root"/}"
    if [[ -n "$prefix" ]]; then
      copy_one "$src" "$prefix/$rel"
    else
      copy_one "$src" "$rel"
    fi
  done < <(
    find "$root" -type f \
      ! -path '*/.git/*' \
      ! -path '*/__pycache__/*' \
      ! -path '*/.pytest_cache/*' \
      -print0 | sort -z
  )
}

install_tree "$repo_root/control_plane" ""
install_tree "$repo_root/hermes_overlay" ""

if [[ "$mode" == "apply" ]]; then
  echo "Backup directory: $backup_dir"
else
  echo "Dry run only. Re-run with --apply to copy files."
fi
