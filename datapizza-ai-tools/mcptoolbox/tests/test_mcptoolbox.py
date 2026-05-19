from unittest.mock import MagicMock, patch

import pytest

from datapizza.tools import Tool
from datapizza.tools.mcptoolbox import MCPToolBoxException, MCPToolBoxTool


# ----------------------- Helpers -----------------------


def _make_fake_tool(name: str, doc: str = "A test tool"):
    """Create a mock that behaves like a ToolboxSyncTool."""
    fake = MagicMock()
    fake.__name__ = name
    fake.__doc__ = doc
    return fake


# ----------------------- Fixtures -----------------------


@pytest.fixture
def mock_client():
    """Patch ToolboxSyncClient so no real server is needed."""
    with patch("datapizza.tools.mcptoolbox.base.ToolboxSyncClient") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def toolbox(mock_client):
    """Return an MCPToolBoxTool wired to the mocked client."""
    return MCPToolBoxTool(toolbox_url="http://fake-server:5000")


@pytest.fixture
def toolbox_with_names(mock_client):
    """MCPToolBoxTool configured with explicit tool_names."""
    return MCPToolBoxTool(
        toolbox_url="http://fake-server:5000",
        tool_names=["search_artists", "get_album"],
    )


@pytest.fixture
def toolbox_with_toolsets(mock_client):
    """MCPToolBoxTool configured with explicit toolset_names."""
    return MCPToolBoxTool(
        toolbox_url="http://fake-server:5000",
        toolset_names=["neo4j-tools"],
    )


@pytest.fixture
def toolbox_mixed(mock_client):
    """MCPToolBoxTool configured with both tool_names and toolset_names."""
    return MCPToolBoxTool(
        toolbox_url="http://fake-server:5000",
        tool_names=["search_artists"],
        toolset_names=["neo4j-tools"],
    )


# ----------------------- Initialization tests -----------------------


class TestInit:
    def test_valid_url(self, mock_client):
        tb = MCPToolBoxTool(toolbox_url="http://localhost:5000")
        assert tb.toolbox_url == "http://localhost:5000"

    def test_url_is_stripped(self, mock_client):
        tb = MCPToolBoxTool(toolbox_url="  http://localhost:5000  ")
        assert tb.toolbox_url == "http://localhost:5000"

    def test_empty_url_raises(self, mock_client):
        with pytest.raises(ValueError, match="non-empty"):
            MCPToolBoxTool(toolbox_url="")

    def test_blank_url_raises(self, mock_client):
        with pytest.raises(ValueError, match="non-empty"):
            MCPToolBoxTool(toolbox_url="   ")

    def test_client_creation_error(self):
        """If the real ToolboxSyncClient fails, MCPToolBoxException is raised."""
        with patch(
            "datapizza.tools.mcptoolbox.base.ToolboxSyncClient",
            side_effect=Exception("connection refused"),
        ):
            with pytest.raises(MCPToolBoxException, match="creating toolbox client"):
                MCPToolBoxTool(toolbox_url="http://unreachable:5000")

    def test_stores_tool_names(self, mock_client):
        names = ["tool_a", "tool_b"]
        tb = MCPToolBoxTool(toolbox_url="http://localhost:5000", tool_names=names)
        assert tb.tool_names == names

    def test_stores_toolset_names(self, mock_client):
        sets = ["set_a"]
        tb = MCPToolBoxTool(toolbox_url="http://localhost:5000", toolset_names=sets)
        assert tb.toolset_names == sets


# ----------------------- _datapizza_tool_wrapper tests -----------------------


class TestWrapper:
    def test_wraps_to_datapizza_tool(self, toolbox):
        fake = _make_fake_tool("my_tool", "Does something useful")
        wrapped = toolbox._datapizza_tool_wrapper(fake)
        assert isinstance(wrapped, Tool)
        assert wrapped.name == "my_tool"
        assert wrapped.description == "Does something useful"


# ----------------------- load_single_tool tests -----------------------


class TestLoadSingleTool:
    def test_returns_tool(self, toolbox, mock_client):
        mock_client.load_tool.return_value = _make_fake_tool("search_artists")
        result = toolbox.load_single_tool("search_artists")
        assert isinstance(result, Tool)
        assert result.name == "search_artists"
        mock_client.load_tool.assert_called_once()

    def test_empty_name_raises_value_error(self, toolbox):
        with pytest.raises(ValueError, match="must not be empty"):
            toolbox.load_single_tool("")

    def test_whitespace_name_raises_value_error(self, toolbox):
        with pytest.raises(ValueError, match="must not be empty"):
            toolbox.load_single_tool("   ")

    def test_not_found_raises_exception(self, toolbox, mock_client):
        mock_client.load_tool.side_effect = ValueError("tool not found")
        with pytest.raises(MCPToolBoxException, match="Can't find tool"):
            toolbox.load_single_tool("nonexistent_tool")

    def test_generic_error_raises_exception(self, toolbox, mock_client):
        mock_client.load_tool.side_effect = RuntimeError("server error")
        with pytest.raises(MCPToolBoxException, match="Unexpected error"):
            toolbox.load_single_tool("broken_tool")

    def test_auth_token_getters_override(self, toolbox, mock_client):
        mock_client.load_tool.return_value = _make_fake_tool("secure_tool")
        custom_auth = {"my_service": lambda: "token123"}
        toolbox.load_single_tool("secure_tool", auth_token_getters=custom_auth)
        _, kwargs = mock_client.load_tool.call_args
        assert kwargs["auth_token_getters"] == custom_auth

    def test_bound_params_override(self, toolbox, mock_client):
        mock_client.load_tool.return_value = _make_fake_tool("param_tool")
        custom_params = {"user_id": "abc"}
        toolbox.load_single_tool("param_tool", bound_params=custom_params)
        _, kwargs = mock_client.load_tool.call_args
        assert kwargs["bound_params"] == custom_params


# ----------------------- load_tools tests -----------------------


class TestLoadTools:
    def test_no_names_loads_default_toolset(self, toolbox, mock_client):
        mock_client.load_toolset.return_value = [
            _make_fake_tool("tool_a"),
            _make_fake_tool("tool_b"),
        ]
        tools = toolbox.load_tools()
        assert len(tools) == 2
        assert all(isinstance(t, Tool) for t in tools)
        mock_client.load_toolset.assert_called_once()

    def test_with_tool_names(self, toolbox_with_names, mock_client):
        mock_client.load_tool.side_effect = [
            _make_fake_tool("search_artists"),
            _make_fake_tool("get_album"),
        ]
        tools = toolbox_with_names.load_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"search_artists", "get_album"}

    def test_with_toolset_names(self, toolbox_with_toolsets, mock_client):
        mock_client.load_toolset.return_value = [
            _make_fake_tool("neo4j_query"),
            _make_fake_tool("neo4j_write"),
        ]
        tools = toolbox_with_toolsets.load_tools()
        assert len(tools) == 2

    def test_mixed_tool_and_toolset(self, toolbox_mixed, mock_client):
        mock_client.load_tool.return_value = _make_fake_tool("search_artists")
        mock_client.load_toolset.return_value = [
            _make_fake_tool("neo4j_query"),
        ]
        tools = toolbox_mixed.load_tools()
        assert len(tools) == 2

    def test_no_tools_raises(self, toolbox, mock_client):
        mock_client.load_toolset.return_value = []
        with pytest.raises(MCPToolBoxException, match="Zero tools loaded"):
            toolbox.load_tools()

    def test_failing_single_tool_is_skipped(self, toolbox_with_names, mock_client):
        """If one tool fails, the other should still load."""
        mock_client.load_tool.side_effect = [
            ValueError("not found"),
            _make_fake_tool("get_album"),
        ]
        # Only 1 out of 2 succeeds, but that's still > 0
        tools = toolbox_with_names.load_tools()
        assert len(tools) == 1
        assert tools[0].name == "get_album"


# ----------------------- list_tools tests -----------------------


class TestListTools:
    def test_returns_metadata_dict(self, toolbox, mock_client):
        mock_client.load_toolset.return_value = [
            _make_fake_tool("tool_a", "Desc A"),
        ]
        info = toolbox.list_tools()
        assert "tool_a" in info
        assert info["tool_a"]["description"] == "Desc A"

    def test_empty_on_error(self, toolbox, mock_client):
        mock_client.load_toolset.side_effect = RuntimeError("boom")
        info = toolbox.list_tools()
        assert info == {}


# ----------------------- close tests -----------------------


class TestClose:
    def test_close_delegates_to_client(self, toolbox, mock_client):
        toolbox.close()
        mock_client.close.assert_called_once()

    def test_close_without_client(self, mock_client):
        tb = MCPToolBoxTool(toolbox_url="http://localhost:5000")
        tb.client = None
        # Should not raise
        tb.close()
