import asyncio
import threading
from typing import Any, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.core.errors import ToolTimeoutError


DEFAULT_STARTUP_TIMEOUT = 30.0
DEFAULT_CALL_TIMEOUT = 30.0


class MCPClient:
    """Blocking wrapper around an MCP stdio server.

    The MCP SDK is asyncio based, but the agent loop (and FastAPI
    sync endpoints) are synchronous, so the session runs on a
    dedicated daemon thread with its own event loop.

    The wrapper is restart-safe: if the background thread dies
    (server crash, startup failure), the next call spawns a new
    thread instead of failing forever.
    """

    def __init__(
        self,
        server_script: str = "app/mcp_server.py",
        call_timeout: float = DEFAULT_CALL_TIMEOUT,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    ):
        self.server_script = server_script
        self.call_timeout = call_timeout
        self.startup_timeout = startup_timeout

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._shutdown = threading.Event()
        self._stopped = threading.Event()

        self._session: ClientSession | None = None

        self._startup_error: Exception | None = None

    def start(self) -> None:
        with self._lock:
            needs_start = (
                self._thread is None
                or not self._thread.is_alive()
                or self._stopped.is_set()
            )

            if needs_start:
                self._shutdown.clear()
                self._stopped.clear()
                self._ready.clear()
                self._startup_error = None

                self._thread = threading.Thread(
                    target=self._run_loop,
                    daemon=True,
                )

                self._thread.start()

        if not self._ready.wait(timeout=self.startup_timeout):
            raise RuntimeError(
                f"MCP client startup timed out after "
                f"{self.startup_timeout}s"
            )

        if self._startup_error is not None:
            raise RuntimeError(
                f"MCP client startup failed: {self._startup_error}"
            )

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._connection_lifecycle())
        except Exception as exc:
            self._startup_error = exc

            if not self._ready.is_set():
                self._ready.set()
        finally:
            if self._loop is loop:
                self._loop = None

            loop.close()
            self._stopped.set()

    async def _connection_lifecycle(self) -> None:
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script],
        )

        async with stdio_client(server_params) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:
                self._session = session

                await session.initialize()

                self._ready.set()

                while not self._shutdown.is_set():
                    await asyncio.sleep(0.05)

                self._session = None

    def _submit(self, coro_factory: Callable[[], Any]) -> Any:
        self.start()

        loop = self._loop

        if loop is None:
            raise RuntimeError("MCP client event loop is not running")

        future = asyncio.run_coroutine_threadsafe(coro_factory(), loop)

        try:
            return future.result(timeout=self.call_timeout)
        except TimeoutError:
            future.cancel()

            raise ToolTimeoutError(
                f"MCP call timed out after {self.call_timeout}s"
            )

    def list_tools(self) -> list[Any]:
        return self._submit(self._list_tools)

    async def _list_tools(self) -> list[Any]:
        if self._session is None:
            raise RuntimeError("MCP session is not ready")

        result = await self._session.list_tools()

        return result.tools

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        return self._submit(
            lambda: self._call_tool(name, arguments)
        )

    async def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        if self._session is None:
            raise RuntimeError("MCP session is not ready")

        result = await self._session.call_tool(
            name,
            arguments=arguments,
        )

        if result.is_error:
            return (
                f"Error: MCP tool '{name}' "
                f"execution failed"
            )

        if result.content:
            return "\n".join(
                getattr(item, "text", str(item))
                for item in result.content
            )

        return str(result)

    def close(self) -> None:
        self._shutdown.set()

        thread = self._thread

        if thread is not None:
            thread.join(timeout=5)

        self._thread = None
        self._loop = None
        self._session = None
        self._stopped.set()
