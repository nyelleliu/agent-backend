import asyncio
import threading
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(
        self,
        server_script: str = "app/mcp_server.py",
    ):
        self.server_script = server_script

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._shutdown = threading.Event()

        self._session: ClientSession | None = None

        self._startup_error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._shutdown.clear()
        self._ready.clear()
        self._startup_error = None

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
        )

        self._thread.start()

        self._ready.wait()

        if self._startup_error is not None:
            raise RuntimeError(
                f"MCP client startup failed: "
                f"{self._startup_error}"
            )

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(
                self._connection_lifecycle()
            )
        except Exception as exc:
            self._startup_error = exc

            if not self._ready.is_set():
                self._ready.set()

        finally:
            self._loop.close()
            self._loop = None

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

    def list_tools(self) -> list[Any]:
        self.start()

        future = asyncio.run_coroutine_threadsafe(
            self._list_tools(),
            self._loop,
        )

        return future.result()

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
        self.start()

        future = asyncio.run_coroutine_threadsafe(
            self._call_tool(
                name,
                arguments,
            ),
            self._loop,
        )

        return future.result()

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
        if self._loop is None:
            return

        self._shutdown.set()

        if self._thread is not None:
            self._thread.join(timeout=5)

        self._thread = None
        self._loop = None
        self._session = None