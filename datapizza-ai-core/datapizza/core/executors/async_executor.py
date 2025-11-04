import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any


class AsyncExecutor:
    """Thread-safe event loop executor for running async code from sync contexts."""

    _singleton_instance = None
    _singleton_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "AsyncExecutor":
        """Get or create the global singleton executor instance."""
        with cls._singleton_lock:
            if cls._singleton_instance is None:
                cls._singleton_instance = cls()
            return cls._singleton_instance

    def __init__(self):
        """Initialize a dedicated event loop"""
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread: threading.Thread = threading.Thread(
            target=self._run_loop, daemon=True
        )
        self._started = threading.Event()
        self._thread.start()
        if not self._started.wait(timeout=5):
            message = "AsyncExecutor failed to start background event loop"
            raise RuntimeError(message)

    def _run_loop(self):
        """Run the event loop"""
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        """
        Run a coroutine in the event loop.

        :param coro: Coroutine to execute
        :param timeout: Optional timeout in seconds
        :return: Result of the coroutine
        :raises TimeoutError: If execution exceeds timeout
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            message = f"Operation timed out after {timeout} seconds"
            raise TimeoutError(message) from e

    def get_loop(self):
        """
        Get the event loop.

        :returns: The event loop
        """
        return self._loop
