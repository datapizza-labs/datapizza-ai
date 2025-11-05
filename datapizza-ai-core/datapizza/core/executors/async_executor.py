import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any


class AsyncExecutor:
    """Thread-safe event loop executor for running async code from sync contexts."""

    _singleton_instance: "AsyncExecutor | None" = None
    _singleton_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "AsyncExecutor":
        """Get or create the global singleton executor instance."""
        with cls._singleton_lock:
            if cls._singleton_instance is None or cls._singleton_instance._closed:
                cls._singleton_instance = cls()
            return cls._singleton_instance

    def __init__(self):
        """Initialize a dedicated event loop."""
        self._runner: asyncio.Runner | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._started = threading.Event()
        self._closed = False
        self._thread.start()
        if not self._started.wait(timeout=5):
            raise RuntimeError("AsyncExecutor failed to start background event loop")

    def _run_loop(self):
        """Run the event loop inside a Runner context."""
        try:
            with asyncio.Runner() as runner:
                self._runner = runner
                self._loop = runner.get_loop()
                self._started.set()
                self._loop.run_forever()
        finally:
            self._closed = True
            self._runner = None
            self._loop = None
            self._started.set()

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        """
        Run a coroutine in the event loop.

        :param coro: Coroutine to execute
        :param timeout: Optional timeout in seconds
        :return: Result of the coroutine
        :raises TimeoutError: If execution exceeds timeout
        """
        loop = self._loop
        if loop is None or self._closed:
            raise RuntimeError("AsyncExecutor event loop is not running")

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            with suppress(concurrent.futures.CancelledError):
                future.result(timeout=0)
            raise TimeoutError(f"Operation timed out after {timeout} seconds") from exc
