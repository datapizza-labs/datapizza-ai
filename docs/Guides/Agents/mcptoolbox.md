# Google MCP Toolbox

[Google MCP Toolbox](https://github.com/googleapis/mcp-toolbox) lets you expose database queries, API calls, and other server-side actions as remote tools. The `datapizza-ai-tools-mcptoolbox` package connects those tools to `datapizza-ai` so your agents can use them natively.

## Installation

```bash
pip install datapizza-ai-tools-mcptoolbox
```

## Quick start

```python
from datapizza.tools.mcptoolbox import MCPToolBoxTool

toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    tool_names=["search_artists", "get_album"],
)

tools = toolbox.load_tools()
```

Every tool returned by `load_tools()` is a standard `Tool` object — you can pass it straight into an agent.

## Use with an Agent

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.mcptoolbox import MCPToolBoxTool

client = OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini")

toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    tool_names=["search_artists"],
)

agent = Agent(
    name="music_agent",
    client=client,
    tools=toolbox.load_tools(),
)

response = agent.run("Find me some jazz artists")
print(response.text)
```

## Loading tools

There are three ways to choose which tools get loaded:

### Specific tools

```python
toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    tool_names=["search_artists", "get_album"],
)
tools = toolbox.load_tools()
```

### Toolsets

```python
toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    toolset_names=["neo4j-tools"],
)
tools = toolbox.load_tools()
```

### All tools (default)

When no `tool_names` or `toolset_names` are provided, every tool on the server is loaded automatically.

```python
toolbox = MCPToolBoxTool(toolbox_url="http://localhost:5000")
tools = toolbox.load_tools()
```

## Load a single tool on demand

```python
tool = toolbox.load_single_tool("search_artists")
```

## List available tools

```python
info = toolbox.list_tools()

for name, meta in info.items():
    print(f"{name}: {meta['description']}")
```

## Authentication

If the remote tools require auth tokens you can provide them at init time or as overrides when loading:

```python
toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    auth_token_getters={"my_auth": lambda: "my-token"},
)

# or override at load time
tools = toolbox.load_tools(
    auth_token_getters={"my_auth": lambda: "another-token"},
)
```

## Bound parameters

Bind fixed values to tool parameters so they are injected automatically:

```python
toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    bound_params={"user_id": "abc-123"},
)
```

## Cleanup

Close the underlying connection when you are done:

```python
toolbox.close()
```

## MCPToolBox vs MCPClient

`datapizza-ai` ships two MCP integrations:

| | **MCPClient** | **MCPToolBoxTool** |
|---|---|---|
| Package | `datapizza-ai` (core) | `datapizza-ai-tools-mcptoolbox` |
| Protocol | Generic MCP (stdio / HTTP) | Google MCP Toolbox |
| Auth | Headers | Token getters + bound params |
| Best for | Any MCP server | Google MCP Toolbox deployments |

Choose `MCPClient` when you need to talk to any MCP-compliant server. Choose `MCPToolBoxTool` when the backend is a Google MCP Toolbox instance.

*(Tip: If you want to quickly test how this works locally with two different sample databases, you can check out this example setup repository: [https://github.com/TommasoTerrin/ToolboxTest](https://github.com/TommasoTerrin/ToolboxTest))*
