# MCPToolBoxTool

```bash
pip install datapizza-ai-tools-mcptoolbox
```

<!-- prettier-ignore -->
::: datapizza.tools.mcptoolbox.MCPToolBoxTool
    options:
        show_source: false

## Overview

The MCPToolBoxTool provides integration with [Google MCP Toolbox](https://github.com/googleapis/mcp-toolbox) servers.
It dynamically loads remote tools and wraps them as native Datapizza-AI `Tool` objects, ready to be used inside agents and pipelines.

## Features

- **Dynamic loading** of remote tools from MCP Toolbox servers
- **Selective loading** via `tool_names` and `toolset_names`
- **Authentication** support through token getters
- **Bound parameters** for injecting fixed values into tool calls
- **Automatic discovery** when no specific tools are requested

## Usage Example

```python
from datapizza.tools.mcptoolbox import MCPToolBoxTool

toolbox = MCPToolBoxTool(
    toolbox_url="http://localhost:5000",
    tool_names=["search_artists"],
)

tools = toolbox.load_tools()
```

## Integration with Agents

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
