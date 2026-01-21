from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from typing import Any

import mcp.types as types
from mcp.client.session import ClientSession, SamplingFnT
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.session import ProgressFnT
from pydantic import AnyUrl

from datapizza.core.executors.async_executor import AsyncExecutor
from datapizza.tools import Tool


class MCPClient:
    """
    Helper for interacting with Model Context Protocol servers.

    Can be used in two modes:

    1. **Stateless mode** (default): Each operation creates a new session.
       Good for HTTP-based MCP servers that don't require persistence.

       ```python
       client = MCPClient(url="https://example.com/mcp")
       tools = client.list_tools()
       # Each tool call creates a new session
       ```

    2. **Persistent mode**: Use as an async context manager to keep the session
       alive across multiple operations. Required for stdio-based servers or
       servers that maintain state.

       ```python
       async with MCPClient(command="uvx", args=["my-mcp-server"]) as client:
           tools = await client.a_list_tools()
           # All tool calls share the same session
           agent = Agent(tools=tools)
           await agent.a_run("do something")
       ```

    Args:
        url: The URL of the MCP server.
        command: The command to run the MCP server.
        headers: The headers to pass to the MCP server.
        args: The arguments to pass to the MCP server.
        env: The environment variables to pass to the MCP server.
        timeout: The timeout for the MCP server.
        sampling_callback: The sampling callback to pass to the MCP server.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        command: str | None = None,
        headers: dict[str, str] | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 30,
        sampling_callback: SamplingFnT | None = None,
    ) -> None:
        self.url = url
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.headers = headers or {}
        self.sampling_callback = sampling_callback

        # Persistent session state
        self._persistent_session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

        if not url and not command:
            raise ValueError("Either url or command must be provided")
        if url and command:
            raise ValueError("Only one of url or command must be provided")

    async def __aenter__(self) -> MCPClient:
        """Enter persistent session mode."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        if self.url:
            read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                streamablehttp_client(
                    self.url,
                    headers=self.headers or None,
                    timeout=self._get_timeout,
                )
            )
        elif self.command:
            server_parameters = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env or None,
            )
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_parameters)
            )
        else:
            raise ValueError("Either url or command must be provided")

        session = await self._exit_stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=self._get_timeout,
                sampling_callback=self.sampling_callback,
            )
        )
        await session.initialize()
        self._persistent_session = session
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit persistent session mode and cleanup resources."""
        self._persistent_session = None
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None

    @property
    def _get_timeout(self) -> timedelta:
        return timedelta(seconds=self.timeout)

    @property
    def is_persistent(self) -> bool:
        """Return True if the client is in persistent session mode."""
        return self._persistent_session is not None

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        """
        Yield an initialized ClientSession.

        If the client is in persistent mode (used as async context manager),
        yields the persistent session. Otherwise, creates a new session
        for this single operation.
        """
        # Use persistent session if available
        if self._persistent_session is not None:
            yield self._persistent_session
            return

        # Otherwise create a new session for this operation
        if self.url:
            async with (
                streamablehttp_client(
                    self.url,
                    headers=self.headers or None,
                    timeout=self._get_timeout,
                ) as (read_stream, write_stream, _),
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=self._get_timeout,
                    sampling_callback=self.sampling_callback,
                ) as session,
            ):
                await session.initialize()
                yield session

        elif self.command:
            server_parameters = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env or None,
            )
            async with (
                stdio_client(server_parameters) as (read_stream, write_stream),
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=self._get_timeout,
                    sampling_callback=self.sampling_callback,
                ) as session,
            ):
                await session.initialize()
                yield session

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        progress_callback: ProgressFnT | None = None,
    ) -> types.CallToolResult:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: The name of the tool to call.
            arguments: The arguments to pass to the tool.
            progress_callback: The progress callback to pass to the tool.

        Returns:
            The result of the tool call.
        """
        async with self._session() as session:
            return await session.call_tool(
                tool_name,
                arguments=arguments or {},
                progress_callback=progress_callback,
            )

    def list_tools(self) -> list[Tool]:
        """
        List the tools available on the MCP server.

        Returns:
            A list of :class:`Tool` objects.
        """
        return AsyncExecutor.get_instance().run(self.a_list_tools(), timeout=10)

    async def a_list_tools(self) -> list[Tool]:
        """
        List the tools available on the MCP server.

        Returns:
            A list of :class:`Tool` objects.
        """
        async with self._session() as session:
            result = await session.list_tools()

        tools: list[Tool] = []
        for mcp_tool in result.tools:
            t_name = mcp_tool.name

            def make_execute_tool(tool_name: str):
                async def execute_tool(**kwargs):
                    result = await self.call_tool(
                        tool_name=tool_name,
                        arguments=kwargs or {},
                    )
                    return result.model_dump_json()

                return execute_tool

            schema = mcp_tool.inputSchema or {}
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            tools.append(
                Tool(
                    func=make_execute_tool(t_name),
                    name=mcp_tool.name,
                    description=mcp_tool.description,
                    properties=properties,
                    required=required,
                    strict=bool(schema.get("additionalProperties") is False),
                )
            )

        return tools

    async def list_resources(self) -> types.ListResourcesResult:
        """
        List the resources available on the MCP server.

        Returns:
            A :class:`types.ListResourcesResult` object.
        """
        async with self._session() as session:
            return await session.list_resources()

    async def list_resource_templates(self) -> types.ListResourceTemplatesResult:
        async with self._session() as session:
            return await session.list_resource_templates()

    async def read_resource(self, resource_uri: AnyUrl) -> types.ReadResourceResult:
        async with self._session() as session:
            return await session.read_resource(resource_uri)

    def list_prompts(self) -> types.ListPromptsResult:
        """
        List the prompts available on the MCP server.

        Returns:
            A :class:`types.ListPromptsResult` object.
        """
        return AsyncExecutor.get_instance().run(self.a_list_prompts(), timeout=10)

    async def a_list_prompts(self) -> types.ListPromptsResult:
        """
        List the prompts available on the MCP server.

        Returns:
            A :class:`types.ListPromptsResult` object.
        """
        async with self._session() as session:
            return await session.list_prompts()

    async def get_prompt(
        self, prompt_name: str, arguments: dict[str, str] | None = None
    ) -> types.GetPromptResult:
        """
        Get a prompt from the MCP server.
        """
        async with self._session() as session:
            return await session.get_prompt(prompt_name, arguments or {})
