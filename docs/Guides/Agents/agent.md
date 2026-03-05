# Build your first agent

The `Agent` class is the core component for creating autonomous AI agents in Datapizza AI. It handles task execution, tool management, memory, and planning.

## Basic Usage

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.memory import Memory
from datapizza.tools import tool

agent = Agent(
    name="my_agent",
    system_prompt="You are a helpful assistant",
    client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    # tools=[],
    # max_steps=10,
    # terminate_on_text=True,  # Terminate execution when the client return a plain text
    # memory=memory,
    # stream=False,
    # planning_interval=0
)

res = agent.run("Hi")
print(res.text)
```


## Use Tools

The above agent is quite basic, so let's make it more functional by adding [**tools**](../../API%20Reference/Type/tool.md).

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.memory import Memory
from datapizza.tools import tool

@tool
def get_weather(location: str, when: str) -> str:
    """Retrieves weather information for a specified location and time."""
    return "25 °C"

agent = Agent(name="weather_agent", tools=[get_weather], client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"))
response = agent.run("What's the weather tomorrow in Milan?")

print(response.text)
# Output:
# Tomorrow in Milan, the temperature will be 25 °C.
```


### tool_choice

You can set the parameter `tool_choice` at invoke time.

The accepted values are: `auto`, `required`, `none`, `required_first`, `list["tool_name"]`


```python
res = master_agent.run(
    task_input="what is the weather in milan?", tool_choice="required_first"
)
```

- `auto`: the model will decide if use a tool or not.
- `required_first`: force to use a tool only at the first step, then auto.
- `required`: force to use a tool at every step.
- `none`: force to not use any tool.



## Core Methods

## Structured output at agent level

You can set `output_cls` on the agent to force every model turn in the agent loop
to request structured output with that schema.

No separate method is needed: use the same `run(...)` and `a_run(...)` APIs.

```python
from pydantic import BaseModel
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient


class Person(BaseModel):
    name: str
    age: int


agent = Agent(
    name="person_extractor",
    client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1-mini"),
    output_cls=Person,
)

result = agent.run('{"name": "Alice", "age": 30}')
person = result.structured_data[0]
print(person.name)
```

If the selected client does not support structured responses, the agent raises a
clear `ValueError` (hard-fail behavior).


### Sync run

`run(task_input: str, tool_choice = "auto", **kwargs) -> str`
Execute a task and return the final result.

```python
result = agent.run("What's the weather like today?")
print(result.text)  # "The weather is sunny with 25°C"
```

### Stream invoke
Stream the agent's execution process, yielding intermediate steps. (Do not stream the single answer)

```python
from datapizza.agents.agent import Agent, StepResult
from datapizza.clients.openai import OpenAIClient
from datapizza.memory import Memory
from datapizza.tools import tool

@tool
def get_weather(location: str, when: str) -> str:
    """Retrieves weather information for a specified location and time."""
    return "25 °C"

agent = Agent(name="weather_agent", tools=[get_weather], client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"))

for step in agent.stream_invoke("What's the weather tomorrow in Milan?"):
    print(f"Step {step.index} starting...")
    print(step.text)
```

### Async run

`a_run(task_input: str, **kwargs) -> str`
Async version of run.

```python
import asyncio

async def main():

    agent = Agent(name="agent", client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"))
    return await agent.a_run("Process this request")


res = asyncio.run(main())
print(res.text)
```

### Async stream invoke
`a_stream_invoke(task_input: str, **kwargs) -> AsyncGenerator[str | StepResult, None]`
Stream the agent's execution process, yielding intermediate steps. (Do not stream the single answer)

```python
from datapizza.agents.agent import Agent
from datapizza.clients.openai import OpenAIClient
import asyncio

async def get_response():
    client = OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini")
    agent = Agent(name= "joke_agent",client=client)
    async for step in agent.a_stream_invoke("tell me a joke"):
        print(f"Step {step.index} starting...")
        print(step.text)

asyncio.run(get_response())
```


## Multi-Agent Communication

An agent can call another ones using `can_call` method


```python
from datapizza.agents.agent import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool

client = OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1")

@tool
def get_weather(city: str) -> str:
    return f""" Monday's weather in {city} is cloudy.
                Tuesday's weather in {city} is rainy.
                Wednesday's weather in {city} is sunny
                Thursday's weather in {city} is cloudy,
                Friday's weather in {city} is rainy,
                Saturday's weather in {city} is sunny
                and Sunday's weather in {city} is cloudy."""

weather_agent = Agent(
    name="weather_expert",
    client=client,
    system_prompt="You are a weather expert. Provide detailed weather information and forecasts.",
    tools=[get_weather]
)

planner_agent = Agent(
    name="planner",
    client=client,
    system_prompt="You are a trip planner. Use weather and analysis info to make recommendations."
)

planner_agent.can_call(weather_agent)

response = planner_agent.run(
    "I need to plan a hiking trip in Seattle next week. Can you help analyze the weather and make recommendations?"
)
print(response.text)
```

Alternatively, you can define a tool that manually calls the agent.
The two solutions are more or less identical.

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool

class MasterAgent(Agent):
    system_prompt="You are a master agent. You can call the weather expert to get weather information."
    name="master_agent"

    @tool
    def call_weather_expert(self, task_to_ask: str) -> str:
        @tool
        def get_weather(city: str) -> str:
            return f""" Monday's weather in {city} is cloudy.
                        Tuesday's weather in {city} is rainy.
                        Wednesday's weather in {city} is sunny
                        Thursday's weather in {city} is cloudy,
                        Friday's weather in {city} is rainy,
                        Saturday's weather in {city} is sunny
                        and Sunday's weather in {city} is cloudy."""

        weather_agent = Agent(
            name="weather_expert",
            client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1"),
            system_prompt="You are a weather expert. Provide detailed weather information and forecasts.",
            tools=[get_weather]
        )
        res = weather_agent.run(task_to_ask)
        return res.text

master_agent = MasterAgent(
    client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1"),
)

master_agent.run("What is the weather in Rome?")
```

### End on agent-as-tool

When converting an agent to a tool with `as_tool`, you can set `end=True` to mark
that tool as terminal.

This is useful for router/orchestrator agents that should stop immediately after
delegating to a sub-agent.

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1")

text_agent = Agent(
    name="text_rag",
    client=client,
    system_prompt="You answer questions using retrieved knowledge.",
)
text_agent_tool = text_agent.as_tool(end=True)

orchestrator = Agent(
    name="orchestrator",
    client=client,
    system_prompt="Route the user request to the best expert.",
    tools=[text_agent_tool],
)

result = orchestrator.run("Help me solve this error in production")
print(result.text)
```

### Tool description in multi-agent

You can set an agent description directly in `Agent(...)`.
When the agent is converted to a tool with `as_tool()`, this description is used
for function calling.

```python
text_agent = Agent(
    name="text_rag",
    description="Answers user questions using company knowledge base retrieval.",
    client=client,
    system_prompt="You answer questions using retrieved knowledge.",
)

tool = text_agent.as_tool(end=True)
```

If `description` is not provided, Datapizza uses this fallback order:

1. agent description from `__init__`
2. agent class docstring
3. agent name


## Planning System

When `planning_interval > 0`, the agent creates execution plans at regular intervals:

During the planning stages, the agent spends time thinking about what the next steps are to be taken to achieve the task.

```python
agent = Agent(
    name="Agent_with_plan",
    client=client,
    planning_interval=3,  # Plan every 3 steps
)
```

The planning system generates structured plans that help the agent organize complex tasks.


## Stream output response

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.core.clients import ClientResponse
from datapizza.tools import tool

client = OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1")

agent = Agent(
    name="Big_boss",
    client=client,
    system_prompt="You are a helpful assistant that answers questions based on the provided context.",
    stream=True, # With stream=True, the agent will stream the client resposne, not only the intermediate steps

)

for r in agent.stream_invoke("What is the weather in Milan?"):
    if isinstance(r, ClientResponse):
        print(r.delta, end="", flush=True)
```
