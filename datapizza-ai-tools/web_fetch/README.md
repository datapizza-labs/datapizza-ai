<div align="center">
<img src="https://github.com/datapizza-labs/datapizza-ai/raw/main/docs/assets/logo_bg_dark.png" alt="Datapizza AI Logo" width="200" height="200">

# Datapizza AI - Web Fetch Tool

**A tool for Datapizza AI that allows agents to fetch and process content from web pages.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

This tool provides a simple and effective way for `datapizza-ai` agents to access information from the internet. Agents equipped with this tool can be instructed to fetch the content of a URL, which they can then use for summarization, data extraction, or to answer questions.

## ⚙️ How it Works

The `WebFetchTool` is a callable class that, once instantiated, can be passed directly to an agent's tool list. The agent will invoke the tool using its registered name, `web_fetch`. It uses the `httpx` library to make a GET request to the given URL and returns the content as a string.

## 🚀 Quick Start

### 1. Installation

```bash
# Install the core framework
pip install datapizza-ai

# Install the WebFetch tool
pip install datapizza-ai-tools-web-fetch
```

### 2. Example: Creating a Web Research Agent

In this example, we'll create an agent that can summarize the content of a web page.

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.web_fetch import WebFetchTool

# 1. Initialize a client (e.g., OpenAI)
client = OpenAIClient(api_key="YOUR_API_KEY")

# 2. Initialize the WebFetchTool
web_tool = WebFetchTool()

# 3. Create an agent and provide it with the web fetch tool
agent = Agent(
    name="WebFetchAgent",
    client=client,
    system_prompt="""You are a helpful research assistant.
Your goal is to answer user questions by fetching information from web pages.

Follow these steps:
1.  Receive a user question that includes a URL.
2.  Use the `web_fetch` tool to get the content of the URL.
3.  Analyze the content and provide a concise summary or answer to the user's question.
""",
    tools=[web_tool]
)

# 4. Run the agent to answer a question
question = "Summarize the main points of the article at https://loremipsum.io/"
print(f"--- Running agent for: '{question}' ---")
response = agent.run(question)
print(f"Agent Response: {response.text}\n")

# Example with a different website
# For this example, we'll stick to example.com to ensure it runs.
question_2 = "What is the title of the page at http://example.com?"
print(f"--- Running agent for: '{question_2}' ---")
response_2 = agent.run(question_2)
print(f"Agent Response: {response_2.text}\n")

```

### Expected Output

The output will vary depending on the live content of the URL. For `https://loremipsum.io/`, it might look something like this:

```
--- Running agent for: 'Summarize the main points of the article at https://loremipsum.io/' ---
Agent Response: The article on **loremipsum.io** provides a comprehensive overview of "Lorem Ipsum," which is a placeholder text commonly used in the graphic, print, and publishing industries. Here are the main points:
```
