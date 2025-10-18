# WebFetch

```bash
pip install datapizza-ai-tools-web-fetch
```

<!-- prettier-ignore -->
::: datapizza.tools.web_fetch.base.WebFetchTool
    options:
        show_source: false

## Overview

The WebFetch tool provides a simple and robust way for AI agents to retrieve content from a given URL. It allows models to access live information from the internet, which is crucial for tasks requiring up-to-date data.

## Features

- **Live Web Access**: Fetches content from any public URL.
- **Error Handling**: Gracefully handles common HTTP errors (e.g., timeouts, 404, 503) and reports them clearly.
- **Configurable**: Allows setting custom timeouts and User-Agent strings.
- **Simple Integration**: As a callable tool, it integrates seamlessly with `datapizza-ai` agents.

## Usage Example

```python
from datapizza.tools.web_fetch import WebFetchTool

# Initialize the tool
fetch_tool = WebFetchTool()

# Fetch content from a URL by calling the tool instance
content = fetch_tool("https://example.com")

print(content)
```

## Integration with Agents

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.web_fetch import WebFetchTool

# 1. Initialize the WebFetchTool, optionally with a custom timeout
web_tool = WebFetchTool(timeout=15.0)

# 2. Create an agent and provide it with the tool
agent = Agent(
    name="web_researcher",
    client=OpenAIClient(api_key="YOUR_API_KEY"),
    system_prompt="You are a research assistant. Use the web_fetch tool to get information from URLs to answer questions.",
    tools=[web_tool]
)

# 3. Run the agent to summarize a web page
response = agent.run("Please summarize the content of https://loremipsum.io/")
print(response.text)
```
