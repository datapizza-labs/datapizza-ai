from datapizza.tools import Tool, tool

from toolbox_core import ToolboxSyncClient
from toolbox_core.sync_tool import ToolboxSyncTool

from typing import Any, Awaitable, Callable, Mapping, Optional, Union


# ----------------------- Exception -----------------------
class MCPToolBoxException(Exception):
    """Exception raised for errors in the MCP Toolbox integration."""

    pass


class MCPToolBoxTool:
    """
    Factory class to load tools from a Google MCP Toolbox service.
    This class handles the connection to an MCP Toolbox server and provides
    methods to retrieve and wrap remote tools into standard Datapizza-AI Tool objects.

    Example:
        ```python
        from datapizza.tools.mcptoolbox import MCPToolBoxTool

        # Initialize the factory
        toolbox = MCPToolBoxTool(
            toolbox_url="http://localhost:5000",
            tool_names=["sql_query", "search_user_sql"],
            toolset_names=["reporting_tools"]
        )

        # Load all configured tools
        agent_tools = toolbox.load_tools()

        # Or load a specific tool on demand
        single_tool = toolbox.load_single_tool("custom_action")
        ```
    """

    def __init__(
        self,
        toolbox_url: str,
        tool_names: list[str] | None = None,
        toolset_names: list[str] | None = None,
        auth_token_getters: Mapping[
            str, Union[Callable[[], str], Callable[[], Awaitable[str]]]
        ] = {},
        client_headers: Optional[
            Mapping[str, Union[Callable[[], str], Callable[[], Awaitable[str]], str]]
        ] = None,
        bound_params: Mapping[
            str, Union[Callable[[], Any], Callable[[], Awaitable[Any]], Any]
        ] = {},
    ):
        """
        Initializes the MCPToolBox factory.

        Args:
            toolbox_url: The base URL of the MCP Toolbox service (e.g., "http://localhost:5000").
            tool_names: Optional list of specific tool names to be loaded via load_tools().
            toolset_names: Optional list of toolset names to be loaded via load_tools().
            auth_token_getters: A mapping of authentication service names to callables that return the tokens.
            client_headers: Headers to include in each request sent through the toolbox client.
            bound_params: A mapping of parameter names to bind to specific values or callables.
        """
        if not toolbox_url or not toolbox_url.strip():
            raise ValueError("toolbox_url must be a non-empty string")
        self.toolbox_url = toolbox_url.strip()
        self.tool_names = tool_names
        self.toolset_names = toolset_names
        self.auth_token_getters = auth_token_getters
        self.client_headers = client_headers
        self.bound_params = bound_params
        self.client = self._create_client()

    # ----------------------- "Private" methods -----------------------
    def _create_client(self) -> ToolboxSyncClient:
        """
        Creates and returns a synchronous Toolbox client.

        Returns:
            ToolboxSyncClient: An initialized synchronous client for the Toolbox server.

        Raises:
            MCPToolBoxException: If there is an error connecting to the server.
        """
        try:
            return ToolboxSyncClient(
                self.toolbox_url, client_headers=self.client_headers
            )
        except ConnectionError as e:
            raise MCPToolBoxException(f"Error connecting to Toolbox server: {e}")
        except Exception as e:
            raise MCPToolBoxException(f"Error creating toolbox client: {e}")

    def close(self):
        """Closes the underlying toolbox client session."""
        if hasattr(self, "client") and self.client:
            self.client.close()

    def _datapizza_tool_wrapper(self, toolbox_tool: ToolboxSyncTool) -> Tool:
        """
        Wraps a ToolboxSyncTool into a Datapizza-AI Tool object.

        Uses the dataclass-based `@tool` decorator logic to preserve the tool's
        metadata, including its name, docstring, and parameter signature.

        Args:
            toolbox_tool: The Google MCP Toolbox tool object to wrap.

        Returns:
            Tool: A standard Datapizza-AI tool object ready to be used by agents.
        """
        return tool(
            func=toolbox_tool,
            name=toolbox_tool.__name__,
            description=toolbox_tool.__doc__,
        )

    # ----------------------- methods for the users -----------------------
    def load_tools(
        self,
        auth_token_getters: Optional[
            Mapping[str, Union[Callable[[], str], Callable[[], Awaitable[str]]]]
        ] = None,
        bound_params: Optional[
            Mapping[str, Union[Callable[[], Any], Callable[[], Awaitable[Any]], Any]]
        ] = None,
    ) -> list[Tool]:
        """
        Loads all configured single tools and toolsets.

        Iterates through the lists provided during initialization and converts
        them into Datapizza-AI Tool objects.

        Args:
            auth_token_getters: Optional override for authentication token getters.
            bound_params: Optional override for bound parameters.

        Returns:
            list[Tool]: A list of initialized Datapizza-AI Tool objects.

        Raises:
            MCPToolBoxException: If no tools could be loaded.
        """
        tools: list[Tool] = []

        # Use provided overrides or class defaults
        auth_getters = (
            auth_token_getters
            if auth_token_getters is not None
            else self.auth_token_getters
        )
        params = bound_params if bound_params is not None else self.bound_params

        # 1. Unified tool list loading logic
        if self.tool_names or self.toolset_names:
            # Load specific tools
            if self.tool_names:
                for tool_name in self.tool_names:
                    try:
                        tools.append(
                            self.load_single_tool(
                                tool_name,
                                auth_token_getters=auth_getters,
                                bound_params=params,
                            )
                        )
                    except MCPToolBoxException as e:
                        print(f"Warning: {e}")

            # Load specific toolsets
            if self.toolset_names:
                for toolset_name in self.toolset_names:
                    try:
                        tb_toolset = self.client.load_toolset(
                            toolset_name,
                            auth_token_getters=auth_getters,
                            bound_params=params,
                        )
                        for tb_tool in tb_toolset:
                            tools.append(self._datapizza_tool_wrapper(tb_tool))
                    except Exception as e:
                        print(f"Warning: Error loading toolset '{toolset_name}': {e}")
        else:
            # 2. Fallback: Load ALL tools from the server if nothing is specified
            try:
                print(
                    "No specific tools/sets requested. Loading default toolset from server."
                )
                tb_all = self.client.load_toolset(
                    name=None, auth_token_getters=auth_getters, bound_params=params
                )
                for tb_tool in tb_all:
                    tools.append(self._datapizza_tool_wrapper(tb_tool))
            except Exception as e:
                print(f"Warning: Error loading all tools from server: {e}")

        if not tools:
            print(
                "Warning: Nessun tool caricato. Controlla la connessione al server o la configurazione."
            )
            raise MCPToolBoxException(
                "Zero tools loaded. Check tool_names, toolset_names, or server availability."
            )

        return tools

    def list_tools(self) -> dict[str, dict[str, Any]]:
        """
        Retrieves detailed metadata for all tools currently accessible through this factory.

        Returns:
            dict: A mapping of tool names to their metadata:
                - description: A short description of the tool.
                - properties: Dictionary of parameter names and their types/schemas.
                - required: List of required parameter names.
        """
        try:
            # Temporarily load all tools to extract metadata
            tools = self.load_tools()

            info_map = {}
            for t in tools:
                info_map[t.name] = {
                    "description": t.description,
                    "properties": t.properties,
                    "required": t.required,
                }
            return info_map
        except Exception as e:
            print(f"Error retrieving tools info: {e}")
            return {}

    def load_single_tool(
        self,
        tool_name: str,
        auth_token_getters: Optional[
            Mapping[str, Union[Callable[[], str], Callable[[], Awaitable[str]]]]
        ] = None,
        bound_params: Optional[
            Mapping[str, Union[Callable[[], Any], Callable[[], Awaitable[Any]], Any]]
        ] = None,
    ) -> Tool:
        """
        Loads a single specific tool from the Toolbox server.

        Args:
            tool_name: The name of the remote tool to load.
            auth_token_getters: Optional override for authentication token getters.
            bound_params: Optional override for bound parameters.

        Returns:
            Tool: The wrapped Datapizza-AI tool.

        Raises:
            ValueError: If tool_name is invalid.
            MCPToolBoxException: If the tool cannot be found or an error occurs.
        """
        if not tool_name or not tool_name.strip():
            raise ValueError("tool_name must not be empty or whitespace")

        # Use provided overrides or class defaults
        auth_getters = (
            auth_token_getters
            if auth_token_getters is not None
            else self.auth_token_getters
        )
        params = bound_params if bound_params is not None else self.bound_params

        try:
            tb_tool = self.client.load_tool(
                tool_name, auth_token_getters=auth_getters, bound_params=params
            )
            return self._datapizza_tool_wrapper(tb_tool)
        except ValueError as e:
            raise MCPToolBoxException(
                f"Can't find tool '{tool_name}' in this server: {e}"
            )
        except Exception as e:
            raise MCPToolBoxException(
                f"Unexpected error loading tool '{tool_name}': {e}"
            )


# Va anche passato nell' __init__ quando sarà fatto
class AsyncMCPToolBoxTool(Tool):
    def __init__():
        pass
