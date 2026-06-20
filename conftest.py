"""Pytest import overlay for the hagent-code checkout.

The repository stores only the files that are overlaid into a full Hermes
checkout. During local tests, import the installed Hermes packages first, then
prefer the checkout's patched subpackages for the files under test.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HERMES_AGENT = Path(
    os.environ.get("HERMES_AGENT")
    or os.environ.get("HERMES_AGENT_HOME")
    or Path.home() / ".hermes" / "hermes-agent"
)


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _prepend_package_path(package: str, path: Path) -> None:
    if not path.exists():
        return
    try:
        module = importlib.import_module(package)
    except ImportError:
        return
    package_path = getattr(module, "__path__", None)
    if package_path is None:
        return
    value = str(path)
    if value not in package_path:
        package_path.insert(0, value)


if HERMES_AGENT.exists():
    _prepend_sys_path(HERMES_AGENT)

_prepend_package_path("gateway", ROOT / "control_plane" / "gateway")
_prepend_package_path(
    "gateway.control_planes",
    ROOT / "control_plane" / "gateway" / "control_planes",
)
_prepend_package_path("agent", ROOT / "hermes_overlay" / "agent")
_prepend_package_path(
    "agent.transports",
    ROOT / "hermes_overlay" / "agent" / "transports",
)

_prepend_sys_path(ROOT / "control_plane")
_prepend_sys_path(ROOT / "hermes_overlay")
