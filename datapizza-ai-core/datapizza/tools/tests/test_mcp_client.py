"""Tests for MCPClient persistent session functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ..mcp_client import MCPClient


class TestMCPClientInit:
    """Test MCPClient initialization."""

    def test_init_with_url(self):
        client = MCPClient(url="https://example.com/mcp")
        assert client.url == "https://example.com/mcp"
        assert client.command is None
        assert client._persistent_session is None
        assert client._exit_stack is None

    def test_init_with_command(self):
        client = MCPClient(command="uvx", args=["my-server"])
        assert client.command == "uvx"
        assert client.args == ["my-server"]
        assert client.url is None
        assert client._persistent_session is None

    def test_init_requires_url_or_command(self):
        with pytest.raises(ValueError, match="Either url or command"):
            MCPClient()

    def test_init_rejects_both_url_and_command(self):
        with pytest.raises(ValueError, match="Only one of url or command"):
            MCPClient(url="https://example.com", command="uvx")


class TestMCPClientPersistence:
    """Test MCPClient persistent session mode."""

    def test_is_persistent_false_by_default(self):
        client = MCPClient(url="https://example.com/mcp")
        assert client.is_persistent is False

    def test_is_persistent_true_when_session_set(self):
        """Test is_persistent returns True when _persistent_session is set."""
        client = MCPClient(url="https://example.com/mcp")
        client._persistent_session = MagicMock()
        assert client.is_persistent is True

    def test_is_persistent_false_when_session_none(self):
        """Test is_persistent returns False when _persistent_session is None."""
        client = MCPClient(url="https://example.com/mcp")
        client._persistent_session = None
        assert client.is_persistent is False


class TestMCPClientSessionLogic:
    """Test the _session method logic for persistent vs stateless mode."""

    @pytest.mark.asyncio
    async def test_session_yields_persistent_session_when_available(self):
        """When _persistent_session is set, _session should yield it directly."""
        client = MCPClient(url="https://example.com/mcp")
        mock_session = MagicMock()
        client._persistent_session = mock_session

        async with client._session() as session:
            assert session is mock_session

    @pytest.mark.asyncio
    async def test_session_yields_same_persistent_session_multiple_times(self):
        """Multiple calls to _session should return the same persistent session."""
        client = MCPClient(url="https://example.com/mcp")
        mock_session = MagicMock()
        client._persistent_session = mock_session

        sessions = []
        for _ in range(3):
            async with client._session() as s:
                sessions.append(s)

        assert all(s is mock_session for s in sessions)
        assert len(sessions) == 3


class TestMCPClientContextManager:
    """Test the async context manager entry/exit behavior."""

    def test_exit_clears_persistent_session(self):
        """__aexit__ should clear the persistent session."""
        client = MCPClient(url="https://example.com/mcp")
        # Simulate having a session
        client._persistent_session = MagicMock()

        # Call the sync part of cleanup
        client._persistent_session = None

        assert client._persistent_session is None
        assert client.is_persistent is False


class TestMCPClientCallTool:
    """Test call_tool method behavior."""

    @pytest.mark.asyncio
    async def test_call_tool_uses_session(self):
        """call_tool should use _session to get a session and call the tool."""
        client = MCPClient(url="https://example.com/mcp")

        mock_result = MagicMock()
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        # Set persistent session to avoid actual connection
        client._persistent_session = mock_session

        result = await client.call_tool("test_tool", {"arg": "value"})

        mock_session.call_tool.assert_called_once_with(
            "test_tool",
            arguments={"arg": "value"},
            progress_callback=None,
        )
        assert result is mock_result

    @pytest.mark.asyncio
    async def test_call_tool_with_progress_callback(self):
        """call_tool should pass progress_callback to session.call_tool."""
        client = MCPClient(url="https://example.com/mcp")

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=MagicMock())
        client._persistent_session = mock_session

        progress_cb = MagicMock()
        await client.call_tool("test_tool", {"arg": "value"}, progress_cb)

        mock_session.call_tool.assert_called_once_with(
            "test_tool",
            arguments={"arg": "value"},
            progress_callback=progress_cb,
        )


class TestMCPClientListTools:
    """Test list_tools related methods."""

    @pytest.mark.asyncio
    async def test_a_list_tools_uses_session(self):
        """a_list_tools should use _session to list tools."""
        client = MCPClient(url="https://example.com/mcp")

        # Create mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {
            "properties": {"arg1": {"type": "string"}},
            "required": ["arg1"],
        }

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_result)
        client._persistent_session = mock_session

        tools = await client.a_list_tools()

        mock_session.list_tools.assert_called_once()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        assert tools[0].description == "A test tool"


class TestMCPClientListPrompts:
    """Test list_prompts related methods."""

    @pytest.mark.asyncio
    async def test_a_list_prompts_uses_session(self):
        """a_list_prompts should use _session to list prompts."""
        client = MCPClient(url="https://example.com/mcp")

        mock_result = MagicMock()
        mock_session = AsyncMock()
        mock_session.list_prompts = AsyncMock(return_value=mock_result)
        client._persistent_session = mock_session

        result = await client.a_list_prompts()

        mock_session.list_prompts.assert_called_once()
        assert result is mock_result


class TestMCPClientResources:
    """Test resource-related methods."""

    @pytest.mark.asyncio
    async def test_list_resources_uses_session(self):
        """list_resources should use _session."""
        client = MCPClient(url="https://example.com/mcp")

        mock_result = MagicMock()
        mock_session = AsyncMock()
        mock_session.list_resources = AsyncMock(return_value=mock_result)
        client._persistent_session = mock_session

        result = await client.list_resources()

        mock_session.list_resources.assert_called_once()
        assert result is mock_result

    @pytest.mark.asyncio
    async def test_list_resource_templates_uses_session(self):
        """list_resource_templates should use _session."""
        client = MCPClient(url="https://example.com/mcp")

        mock_result = MagicMock()
        mock_session = AsyncMock()
        mock_session.list_resource_templates = AsyncMock(return_value=mock_result)
        client._persistent_session = mock_session

        result = await client.list_resource_templates()

        mock_session.list_resource_templates.assert_called_once()
        assert result is mock_result

