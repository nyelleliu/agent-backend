import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.mcp.registry import MCPToolRegistry


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["app/mcp_server.py"],
    )

    async with stdio_client(server_params) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:

            await session.initialize()

            registry = MCPToolRegistry(session)

            await registry.refresh()

            print("Available MCP tools:")

            for schema in registry.schemas():
                print(schema)

            print("\nMCP tool discovery succeeded.")


if __name__ == "__main__":
    asyncio.run(main())
