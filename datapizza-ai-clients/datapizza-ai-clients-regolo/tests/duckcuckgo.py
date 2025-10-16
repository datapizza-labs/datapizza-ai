from datapizza.agents import Agent
from datapizza.clients.regolo import RegoloClient
from datapizza.tools.duckduckgo import DuckDuckGoSearchTool
from datapizza.tracing import ContextTracing

import os

# Get API key from environment or raise error if not set
api_key = os.getenv("REGOLO_API_KEY")
if not api_key:
    raise ValueError(
        "REGOLO_API_KEY environment variable not set. "
        "Please set it with: export REGOLO_API_KEY='your-api-key'"
    )

client = RegoloClient(api_key=api_key, model="mistral-small3.2")
agent = Agent(
    name="assistant",
    client=client,
    tools=[DuckDuckGoSearchTool()],
    max_steps=3,
)

with ContextTracing().trace("my_ai_operation"):
    response = agent.run("Tell me some news about Bitcoin")