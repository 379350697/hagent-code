"""Async bridge for blocking Codex app-server calls."""

from __future__ import annotations

import asyncio
import threading
from typing import Callable, TypeVar

T = TypeVar("T")


async def run_blocking(func: Callable[..., T], *args) -> T:
    """Run a blocking function in a dedicated daemon thread.

    The gateway cannot rely on ``asyncio.to_thread`` in every deployment: some
    embedded runner/plugin combinations leave the default executor unresolved.
    A tiny explicit thread bridge keeps Codex turns off the adapter event loop.
    """

    result_box: dict[str, object] = {}

    def worker() -> None:
        try:
            result_box["value"] = func(*args)
        except BaseException as exc:
            result_box["error"] = exc

    thread = threading.Thread(target=worker, name="codex-command-worker", daemon=True)
    thread.start()
    while thread.is_alive():
        await asyncio.sleep(0.05)
    if "error" in result_box:
        raise result_box["error"]  # type: ignore[misc]
    return result_box.get("value")  # type: ignore[return-value]
