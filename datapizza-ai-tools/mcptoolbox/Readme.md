# MCPToolBox - Datapizza-AI Integration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Integration module for **Google MCP Toolbox** within the **Datapizza-AI** framework.

This module allows you to dynamically load remote tools exposed via an MCP Toolbox server and use them as native Datapizza-AI tools within agents and pipelines.

---

## 🎯 Key Features

- ✅ **Dynamic loading** of remote tools from MCP Toolbox servers
- ✅ **Zero boilerplate**: no manual wrappers needed
- ✅ **Type-safe**: preserves type hints from remote tools
- ✅ **Authentication support** and bound parameters
- ✅ **Auto-discovery**: automatic loading of all available tools
- ✅ **Robust error handling** with graceful degradation
- ✅ **Seamless integration** with the Datapizza-AI framework

---

## 📦 Installation

### Via UV (recommended)

If you're using the complete Datapizza-AI workspace:

```bash
uv pip install -e ".[all]"
```

### Standalone installation

```bash
uv pip install datapizza-ai-tools-mcptoolbox
```

### Dependencies

The module requires:
- `datapizza-ai-core`
- `toolbox-core>=0.5.8`

---

## 🚀 Quick Start

### Basic Example

```python
from datapizza.tools.mcptoolbox import MCPToolBoxTool

# 1. Initialize the factory by connecting to the MCP Toolbox server
toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    tool_names=["sql_query", "search_user"],
    toolset_names=["reporting_tools"]
)

# 2. Load the configured tools
tools = toolbox.load_tools()

print(f"Loaded {len(tools)} tools")

# 3. Use the tools with a Datapizza-AI agent
from datapizza.agents import Agent

agent = Agent(tools=tools)
result = agent.run("Find all users who signed up last month")

# 4. Cleanup
toolbox.close()
```

### Auto-Discovery (load all available tools)

If you don't specify `tool_names` or `toolset_names`, the module automatically loads **all available tools** from the server:

```python
toolbox = MCPToolBoxTool(toolbox_url="http://localhost:5000")
all_tools = toolbox.load_tools()
print(f"Discovered {len(all_tools)} tools automatically")
```

---

## 📖 Usage Guide

### Initialization

The `MCPToolBoxTool` class accepts the following parameters:

```python
MCPToolBoxTool(
    toolbox_url: str,                    # MCP Toolbox server URL (required)
    tool_names: list[str] | None = None, # List of specific tools to load
    toolset_names: list[str] | None = None, # List of toolsets to load
    auth_token_getters: Mapping[str, Callable] = {}, # Authentication management
    client_headers: Mapping[str, str] | None = None, # Custom HTTP headers
    bound_params: Mapping[str, Any] = {}  # Pre-bound parameters for tools
)
```

#### Parameters

- **`toolbox_url`** (required): URL of the MCP Toolbox server
- **`tool_names`**: List of specific tool names to load
- **`toolset_names`**: List of toolset names to load (a toolset can contain multiple tools)
- **`auth_token_getters`**: Mapping of authentication service names to callables that return tokens
- **`client_headers`**: Custom HTTP headers to include in each request
- **`bound_params`**: Pre-configured parameters to automatically pass to tools

---

### Main Methods

#### `load_tools(auth_token_getters=None, bound_params=None) -> list[Tool]`

Loads all configured tools and returns them as a list of Datapizza-AI `Tool` objects.

**Behavior:**
- If `tool_names` and/or `toolset_names` are specified, loads only those
- If both are `None`, loads **all available tools** from the server
- Handles errors gracefully: prints warnings but continues loading other tools
- Raises `MCPToolBoxException` if no tools are loaded

**Example:**
```python
tools = toolbox.load_tools()
```

**Override authentication/parameters:**
```python
tools = toolbox.load_tools(
    auth_token_getters={"api": lambda: "new-token"},
    bound_params={"env": "staging"}
)
```

---

#### `load_single_tool(tool_name: str, auth_token_getters=None, bound_params=None) -> Tool`

Loads a single specific tool on-demand.

**Example:**
```python
search_tool = toolbox.load_single_tool("search_artists")
result = search_tool(query="Beatles")
```

**Raises:**
- `ValueError`: If `tool_name` is empty or whitespace
- `MCPToolBoxException`: If the tool is not found or an error occurs

---

#### `list_tools() -> dict[str, dict[str, Any]]`

Returns metadata for all available tools.

**Example:**
```python
tools_info = toolbox.list_tools()

for name, info in tools_info.items():
    print(f"Tool: {name}")
    print(f"  Description: {info['description']}")
    print(f"  Required params: {info['required']}")
    print(f"  All params: {info['properties'].keys()}")
```

**Output:**
```python
{
    "search_artists": {
        "description": "Search for artists in the database",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"}
        },
        "required": ["query"]
    }
}
```

---

#### `close()`

Closes the connection to the MCP Toolbox client, freeing resources.

**Example:**
```python
toolbox.close()
```

**Best practice:** Always use `close()` when finished, or use a context manager (if implemented).

---

## 🔐 Authentication

### Example with Static Token

```python
def get_api_token():
    return "my-secret-token"

toolbox = MCPToolBoxTool(
    toolbox_url="http://api.example.com",
    auth_token_getters={"api_service": get_api_token}
)

tools = toolbox.load_tools()
```

### Example with Dynamic Token

```python
import os

def get_dynamic_token():
    # Retrieve token from environment variable or service
    return os.getenv("API_TOKEN")

toolbox = MCPToolBoxTool(
    toolbox_url="http://api.example.com",
    auth_token_getters={
        "service_a": get_dynamic_token,
        "service_b": lambda: "another-token"
    }
)
```

### Custom HTTP Headers

```python
toolbox = MCPToolBoxTool(
    toolbox_url="http://api.example.com",
    client_headers={
        "X-Custom-Header": "value",
        "X-API-Version": "v2"
    }
)
```

---

## 🔗 Bound Parameters

Bound parameters allow you to pre-configure values that will be automatically passed to all tools.

### Example

```python
toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    bound_params={
        "database": "production",
        "max_results": 100,
        "timeout": 30
    }
)

# These parameters will be automatically passed to all tools
tools = toolbox.load_tools()
```

### Dynamic Parameters

```python
def get_current_user():
    return "user@example.com"

toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    bound_params={
        "user": get_current_user,  # Callable
        "env": "production"         # Static value
    }
)
```

---

## 🧪 Testing

### Running Tests

The module includes comprehensive integration tests:

```bash
# Make sure the MCP Toolbox server is running on http://localhost:5000
python tests/test.py
```

### Included Tests

The tests verify:

1. **Client creation** with different configurations
2. **Single tool loading** with `load_single_tool()`
3. **Auto-discovery** with `load_tools()`
4. **Tool metadata** with `list_tools()`
5. **Resource cleanup** with `close()`

### Example Test Output

```
---- Create 3 agents initializing MCPToolBoxTool with URL: http://localhost:5000 ----

---- Agent 1: created ----
---- Agent 2: created ----
---- Agent 3: created ----

--- Test 1: load_single_tool('search_artists') ---
Type: <class 'datapizza.tools.Tool'>
Name: search_artists
Description: Search for artists in the music database

--- Test 3: load_tools() (Automatic discovery) ---
Discovered 5 tools automatically for client 1.
Discovered 1 tools automatically for client 2.
Discovered 3 tools automatically for client 3.

--- Test 4: info() for client 1 ---

Tool: search_artists
 - Description: Search for artists in the music database
 - Required Params: ['query']

Tool: get_album_reviews
 - Description: Get reviews for a specific album
 - Required Params: ['album_id']

...

--- Cleaning up resources ---
--- Testing complete ---
```

---

## 🛠️ Advanced Examples

### Example 1: Selective Loading

```python
# Load only specific tools
toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    tool_names=["search_artists", "get_album_info"]
)

tools = toolbox.load_tools()
```

### Example 2: Loading Toolsets

```python
# Load all tools from a toolset
toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    toolset_names=["neo4j-tools", "reporting-tools"]
)

tools = toolbox.load_tools()
```

### Example 3: Combining Tools + Toolsets

```python
# Load both individual tools and toolsets
toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    tool_names=["custom_action"],
    toolset_names=["analytics-tools"]
)

tools = toolbox.load_tools()
```

### Example 4: Using with Datapizza-AI Agent

```python
from datapizza.tools.mcptoolbox import MCPToolBoxTool
from datapizza.agents import Agent
from datapizza.llms import OpenAI

# Setup toolbox
toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    toolset_names=["database-tools"]
)

# Load tools
tools = toolbox.load_tools()

# Create agent
agent = Agent(
    llm=OpenAI(model="gpt-4"),
    tools=tools,
    system_prompt="You are a helpful data analyst assistant."
)

# Execute query
response = agent.run("Show me the top 10 customers by revenue")
print(response)

# Cleanup
toolbox.close()
```

---

## ⚠️ Error Handling

### Custom Exception: `MCPToolBoxException`

The module defines a custom exception for specific errors:

```python
from datapizza.tools.mcptoolbox import MCPToolBoxException

try:
    toolbox = MCPToolBoxTool(toolbox_url="http://invalid-url")
    tools = toolbox.load_tools()
except MCPToolBoxException as e:
    print(f"MCP Toolbox error: {e}")
```

### Graceful Error Handling

The `load_tools()` method handles errors gracefully:

```python
toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    tool_names=["existing_tool", "non_existing_tool", "another_tool"]
)

# Prints warning for "non_existing_tool" but loads the others
tools = toolbox.load_tools()
# Output: ⚠️ Warning: Can't find tool 'non_existing_tool' in this server
```

---

## 🔍 Troubleshooting

### Issue: "Error connecting to Toolbox server"

**Cause:** The MCP Toolbox server is not reachable.

**Solution:**
1. Verify the server is running
2. Check that the URL is correct
3. Verify network connectivity

```bash
# Test connection
curl http://localhost:5000/health
```

### Issue: "Zero tools loaded"

**Cause:** No tools were loaded (incorrect names or empty server).

**Solution:**
1. Verify tool names with `list_tools()`
2. Check the MCP Toolbox server configuration
3. Try auto-discovery without specifying names

```python
# Check available tools
toolbox = MCPToolBoxTool(toolbox_url="http://localhost:5000")
print(toolbox.list_tools())
```

### Issue: "Can't find tool 'xyz' in this server"

**Cause:** The specified tool doesn't exist on the server.

**Solution:**
1. Use `list_tools()` to see available tools
2. Verify the server configuration (`tools.yaml` file)

---

## 📋 Requirements

- **Python**: 3.10+
- **Datapizza-AI Core**: Any compatible version
- **Toolbox Core**: >= 0.5.8
- **MCP Toolbox Server**: Running and accessible

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Current Limitations

- **`AsyncMCPToolBoxTool`**: Not yet implemented (placeholder for future versions)
- **Context Manager**: Not yet supported (use `close()` manually)
- **Caching**: Tools are reloaded on each call to `load_tools()`

---

## 🗺️ Roadmap

- [ ] Full implementation of `AsyncMCPToolBoxTool`
- [ ] Context manager support (`with` statement)
- [ ] Intelligent caching of loaded tools
- [ ] Hot-reload of tools without recreating the client
- [ ] Advanced metrics and logging
- [ ] Automatic retry for network errors

---

## 📚 Resources

- [Google MCP Toolbox](https://github.com/google/toolbox-core)
- [Datapizza-AI Documentation](https://docs.datapizza.ai)
- [Datapizza-AI GitHub](https://github.com/datapizza/datapizza-ai)

---

## 📄 License

This module is distributed under the MIT License. See the `LICENSE` file for details.

---

## ✍️ Authors

**Datapizza TEAM** - [team@datapizza.it](mailto:team@datapizza.it)

---

## 🙏 Acknowledgments

- Google team for the MCP Toolbox framework
- Datapizza-AI community for continuous support
- All contributors who made this project possible
