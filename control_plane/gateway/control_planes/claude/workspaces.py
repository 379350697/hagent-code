"""Workspace discovery and per-chat workspace selection for Claude tasks."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is unavailable on Windows.
    fcntl = None


JsonDict = dict[str, Any]
_SKIP_DIRS = {
    ".cache",
    ".claude",
    ".git",
    ".hermes",
    ".mypy_cache",
    ".nox",
    ".nvm",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
}


@dataclass(frozen=True)
class WorkspaceEntry:
    path: str
    name: str
    root: str


def workspace_store_path() -> str:
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(home, "claude-control-plane", "workspaces.json")


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in paths:
        if not raw:
            continue
        path = os.path.abspath(os.path.expanduser(raw))
        if path in seen or not os.path.isdir(path):
            continue
        seen.add(path)
        result.append(path)
    return result


def workspace_scan_roots(default_workspace: str = "") -> list[str]:
    env_value = (
        os.environ.get("HERMES_CLAUDE_WORKSPACE_ROOTS")
        or os.environ.get("CLAUDE_WORKSPACE_ROOTS")
        or ""
    )
    roots = [item.strip() for item in env_value.split(os.pathsep) if item.strip()]
    if not roots:
        roots = [
            default_workspace,
            os.environ.get("HERMES_HOME") or "",
            os.path.expanduser("~"),
        ]
    return _dedupe_paths(roots)


def discover_git_workspaces(
    roots: list[str],
    *,
    max_depth: int = 4,
    limit: int = 80,
) -> list[WorkspaceEntry]:
    entries: list[WorkspaceEntry] = []
    seen: set[str] = set()
    for root in _dedupe_paths(roots):
        root_path = Path(root)
        base_depth = len(root_path.parts)
        for current, dirs, _files in os.walk(root):
            current_path = Path(current)
            depth = max(0, len(current_path.parts) - base_depth)
            dirs[:] = [
                item
                for item in dirs
                if item not in _SKIP_DIRS and not item.startswith(".pytest")
            ]
            if (current_path / ".git").exists():
                if current_path.name.startswith("."):
                    dirs[:] = []
                    continue
                resolved = str(current_path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    entries.append(
                        WorkspaceEntry(
                            path=resolved,
                            name=current_path.name or resolved,
                            root=str(root_path.resolve()),
                        )
                    )
                    if len(entries) >= limit:
                        return sorted(entries, key=lambda item: item.path)
                dirs[:] = []
                continue
            if depth >= max_depth:
                dirs[:] = []
    return sorted(entries, key=lambda item: item.path)


class WorkspaceSelectionStore:
    """Persistent per task-key workspace selection."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or workspace_store_path()
        self._lock = threading.RLock()

    def get(self, task_key: str) -> str:
        with self._lock:
            data = self._load()
        value = data.get(task_key)
        if isinstance(value, dict):
            return str(value.get("workspace") or "")
        return ""

    def set(self, task_key: str, workspace: str) -> str:
        resolved = os.path.abspath(os.path.expanduser(workspace))
        with self._lock:
            data = self._load()
            data[task_key] = {"workspace": resolved, "updatedAt": time.time()}
            self._save(data)
        return resolved

    def clear(self, task_key: str) -> None:
        with self._lock:
            data = self._load()
            data.pop(task_key, None)
            self._save(data)

    @contextmanager
    def _file_lock(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        lock_path = f"{self.path}.lock"
        with open(lock_path, "a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load(self) -> JsonDict:
        with self._file_lock():
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except FileNotFoundError:
                return {}
            except Exception:
                return {}
            selections = payload.get("selections") if isinstance(payload, dict) else {}
            return selections if isinstance(selections, dict) else {}

    def _save(self, selections: JsonDict) -> None:
        with self._file_lock():
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(
                    {"version": 1, "selections": selections},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            os.replace(tmp, self.path)
