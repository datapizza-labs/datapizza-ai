# datapizza-ai

**Build reliable Gen AI solutions without overhead**

`datapizza-ai` provides clear interfaces and predictable behavior for agents and RAG. End-to-end visibility and reliable orchestration keep engineers in control from PoC to scale

## Installation

Install the library using pip:

```bash
pip install datapizza-ai
```

## Key Features

- **Integration with AI Providers**: Seamlessly connect with AI services like OpenAI and Google VertexAI.
- **Complex workflows, minimal code.**: Design, automate, and scale powerful agent workflows without the overhead of boilerplate.
- **Retrieval-Augmented Generation (RAG)**: Enhance AI responses with document retrieval.
- **Faster delivery, easier onboarding for new engineers**: Rebuild a RAG + tools agent without multi-class plumbing; parity with simpler, typed interfaces.
- **Up to 40% less debugging time**: Trace and log every LLM/tool call with inputs/outputs

## Quick Start

To get started with `datapizza-ai`, ensure you have Python `>=3.10.0,<3.13.0` installed.

Here's a basic example demonstrating how to use agents in `datapizza-ai`:

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool

@tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

client = OpenAIClient(api_key="YOUR_API_KEY")
agent = Agent(name="assistant", client=client, tools = [get_weather])

response = agent.run("What is the weather in Rome?")
# output: The weather in Rome is sunny
```
