# Model Context Protocol (MCP)

Model Context Protocol (MCP) is an open-source standard that enables AI applications to connect with external systems like databases, APIs, and tools.

Use MCP (Model Context Protocol) tools inside `datapizza-ai` by wrapping them as
regular agent tools. Follow this minimal recipe to get an agent talking to a
remote MCP server in just a few steps.


With MCP, you can build AI agents that:

- **Access your codebase**: Let AI read GitHub repositories, create issues, and manage pull requests
- **Query your database**: Enable natural language queries against PostgreSQL, MySQL, or any database
- **Browse the web**: Give AI the ability to search and extract information from websites
- **Control your tools**: Connect to Slack, Notion, Google Calendar, or any API-based service
- **Analyze your data**: Let AI work with spreadsheets, documents, and business intelligence tools

## Fetch MCP tools

Here an example of [FastMCP](https://gofastmcp.com/getting-started/welcome) tool provided by FastMCP

```python
from datapizza.tools.mcp_client import MCPClient

fastmcp_client = MCPClient(url="https://gofastmcp.com/mcp")
fastmcp_tools = fastmcp_client.list_tools()
```

## Create the agent and run it

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(api_key="OPENAI_API_KEY", model="gpt-4o-mini")

agent = Agent(
    name="mcp_agent",
    client=client,
    tools=fastmcp_tools,
)

result = agent.run("How can I use a FastMCP server over HTTP?")
print(result.text)
```

That’s it—you now have an agent that discovers tools from the FastMCP server and
uses them as part of normal `datapizza-ai` reasoning. Swap in any MCP endpoint
or different LLM client to match your project.
