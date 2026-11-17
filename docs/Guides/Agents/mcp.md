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

## Return a documentation answer with source URLs

The runnable [Parallel search example](https://github.com/datapizza-labs/datapizza-ai/blob/main/datapizza-ai-core/examples/parallel_search.py)
uses the same `MCPClient` and `Agent` APIs to answer one public question about
Python's `asyncio.TaskGroup`. It discovers the tools at
`https://mcp.parallel.ai/mcp`, selects `web_search` by name, and passes only that
tool to the agent. Parallel search is opt-in for this example.

From a repository checkout, install the core and client, then run:

```bash
python -m pip install -e ./datapizza-ai-core -e ./datapizza-ai-clients/datapizza-ai-clients-openai 'mcp==1.20.0' 'pydantic==2.12.4' 'pydantic-settings==2.12.0'
export OPENAI_API_KEY='your-provider-key'
export OPENAI_MODEL='your-tool-and-structured-output-capable-model'
python datapizza-ai-core/examples/parallel_search.py
```

The MCP and Pydantic versions pin a compatible set for the existing
Streamable HTTP client. The model provider must support tools and structured
output. You can also call `answer_question(client)` with
another compatible Datapizza client.

The example requires a search on the first agent step. Once the agent finishes,
it uses `client.a_structured_response(...)` without tools to format the sourced
answer as a `DocsAnswer`, printed as JSON with `answer` and `sources` fields.
This keeps remote tool schemas out of the structured-output request.
The prompts ask for original URLs from search results, and a local validator
checks their HTTP(S) URL format without rewriting them. Inspect the cited pages
when assessing the answer. If search
returns no useful evidence or a tool error, the prompt asks for an unavailable
answer with an empty `sources` list. Connection and timeout errors propagate to
the caller; a missing `web_search` tool or missing structured answer raises a
`RuntimeError`.

Running the example authorizes the fixed public documentation query to be sent
to Parallel and the question and retrieved excerpts to your model provider.
Keep private inputs out of the query. The provider key is used by the model
client only. Parallel's keyless access has lower rate limits; see
[Parallel MCP setup](https://docs.parallel.ai/integrations/mcp/quickstart).
