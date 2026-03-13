# Build your first agent

Agents are the core building block in Datapizza AI.

An `Agent` combines:

- a `name`
- a `system_prompt`
- a `client`
- optional tools, memory, hooks, structured output, and handoffs

Use this page to get an agent running quickly, then add the capabilities you need.

## Create your first agent

Start with the smallest useful setup.

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient

agent = Agent(
    name="assistant",
    system_prompt="You are a helpful assistant.",
    client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
)
```

The only required pieces are:

- `name`: a human-readable name for the agent
- `system_prompt`: the instructions the model follows
- `client`: the model provider implementation

## Run your first agent

Call `run(...)` and read the final answer from `result.text`.

```python
result = agent.run("Write a one-line welcome message for a new user.")
print(result.text)
```

`run(...)` returns a `StepResult`, not a plain string.

The most useful properties are:

- `result.text`: the final text answer
- `result.tools_used`: tools called in that step
- `result.structured_data`: parsed structured output when `output_cls` is set
- `result.usage`: token usage aggregated for the run

## Give your agent tools

Tools let the agent fetch data or perform actions.

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool


@tool
def get_weather(location: str, when: str) -> str:
    """Return weather information for a location and time."""
    return "25 C"


agent = Agent(
    name="weather_agent",
    system_prompt="You help users with weather questions.",
    client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    tools=[get_weather],
)

result = agent.run("What's the weather tomorrow in Milan?")
print(result.text)
```

### Control tool use

At run time, you can control how the model uses tools with `tool_choice`.

```python
result = agent.run(
    "What's the weather in Milan?",
    tool_choice="required_first",
)
```

Supported values:

- `"auto"`: the model decides whether to use a tool
- `"required"`: the model must use a tool every step
- `"none"`: the model must not use tools
- `"required_first"`: the first step must use a tool, later steps go back to `auto`
- `list[str]`: restrict tool use to a named subset

## Keep memory between runs

You can pass a custom `Memory` object.

```python
from datapizza.memory import Memory

memory = Memory()

agent = Agent(
    name="assistant",
    system_prompt="You are a helpful assistant.",
    client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    stateless=False,
    memory=memory,
)
```

## Stream responses

Use `stream_invoke(...)` when you want to observe the run as it happens.

It yields:

- `ClientResponse` chunks when client streaming is enabled
- `StepResult` objects for completed agent steps
- `Plan` objects when planning is enabled

```python
from datapizza.agents import Agent, StepResult
from datapizza.clients.openai import OpenAIClient
from datapizza.core.clients import ClientResponse

agent = Agent(
    name="assistant",
    system_prompt="You are a helpful assistant.",
    client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    stream=True,
)

for event in agent.stream_invoke("Tell me a short joke."):
    if isinstance(event, ClientResponse):
        print(event.delta, end="", flush=True)
    elif isinstance(event, StepResult):
        print("\nfinal step:", event.text)
```

Async streaming works the same way with `a_stream_invoke(...)`.

```python
import asyncio


async def main():
    agent = Agent(
        name="assistant",
        system_prompt="You are a helpful assistant.",
        client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
        stream=True,
    )

    async for event in agent.a_stream_invoke("Tell me a short joke."):
        print(event)


asyncio.run(main())
```

## Return structured data

If you want typed output instead of plain text, set `output_cls`.

```python
from pydantic import BaseModel
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient


class Person(BaseModel):
    name: str
    age: int


agent = Agent(
    name="person_extractor",
    system_prompt="Extract a person from the input.",
    client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1-mini"),
    output_cls=Person,
)

result = agent.run('{"name": "Alice", "age": 30}')
person = result.structured_data[0]
print(person.name)
```

When `output_cls` is set:

- the agent asks the client for structured output on each model turn
- the parsed objects are available in `result.structured_data`
- `result.text` may be empty

If the selected client does not support structured output, Datapizza raises a clear `ValueError`.

## Choose a multi-agent pattern

Before adding more agents, decide who should own the final answer.

- `can_call(...)` / `as_tool()`: one orchestrator stays in control and calls specialists as tools
- `handoffs`: control transfers to another agent, which continues the run

Use `can_call(...)` when you want a manager pattern.
Use `handoffs` when you want a specialist to take over.

### Agents as tools

In this pattern, the main agent keeps control of the conversation.

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool

client = OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1")


@tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


weather_agent = Agent(
    name="weather_expert",
    description="Answers weather questions.",
    system_prompt="You are a weather expert.",
    client=client,
    tools=[get_weather],
)

planner_agent = Agent(
    name="planner",
    system_prompt="You are a travel planner. Use specialist tools when useful.",
    client=client,
)

planner_agent.can_call(weather_agent)

result = planner_agent.run("Can I go hiking in Milan tomorrow?")
print(result.text)
```

You can also convert an agent manually with `as_tool()`.

```python
tool = weather_agent.as_tool()
```

If delegating should end the orchestrator run immediately, use `end=True`.

```python
terminal_tool = weather_agent.as_tool(end=True)
```

When Datapizza builds a tool from an agent, the tool description is chosen in this order:

1. `description` passed to `Agent(...)`
2. the agent class docstring
3. the agent name

### Handoffs

In this pattern, one agent transfers control to another.

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1")

refund_agent = Agent(
    name="refund_specialist",
    system_prompt="Handle refund requests clearly and safely.",
    client=client,
)

triage_agent = Agent(
    name="triage",
    system_prompt="Route the user to the right specialist.",
    client=client,
    handoffs=[refund_agent],
)

result = triage_agent.run("I was charged twice. I need a refund.")
print(result.text)
```

You can also register handoffs later:

```python
triage_agent.can_handoff(refund_agent)
```

For most users, `agent.run(...)` is enough. Datapizza creates an `AgentRunner` internally.

If you need richer orchestration metadata, use `AgentRunner` directly.

```python
from datapizza.agents import AgentRunner

runner = AgentRunner()
result = runner.run(triage_agent, "I was charged twice.")

print(result.final_step.text)
print(result.final_agent.name)
```

`AgentRunner.run(...)` returns an `AgentRunnerResult` with these fields:

- `final_step`: the final `StepResult`
- `final_agent`: the agent that produced the final answer
- `handoff_count`: how many handoffs happened during the run
- `visited_agents`: the sequence of agents visited during the run
- `memory`: the shared `Memory` used for the run
- `usage`: aggregated `TokenUsage` for the whole run

## Observe the agent loop

Use hooks when you want to log or inspect each step.

```python
from datapizza.agents import Agent, AgentHooks, StepContext, StepResult
from datapizza.clients.openai import OpenAIClient


class DebugHooks(AgentHooks):
    def before_step(self, context: StepContext) -> None:
        print(f"starting step {context.step_index}")

    def after_step(self, context: StepContext, result: StepResult) -> None:
        print(f"finished step {context.step_index}: {result.text}")


agent = Agent(
    name="assistant",
    system_prompt="You are a helpful assistant.",
    client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    hooks=DebugHooks(),
)
```

`before_step(...)` runs at the start of each loop iteration.
`after_step(...)` runs after the step result is produced.

## Plan before acting

If you want the agent to periodically create a plan before continuing, set `planning_interval`.

```python
agent = Agent(
    name="planner_agent",
    system_prompt="You solve tasks carefully.",
    client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1"),
    planning_interval=3,
)
```

When planning is enabled, the agent generates a structured `Plan` at regular intervals and then continues execution.

## Async run

If your application is async, use `a_run(...)`.

```python
import asyncio
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient


async def main():
    agent = Agent(
        name="assistant",
        system_prompt="You are a helpful assistant.",
        client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    )
    result = await agent.a_run("Summarize this text in one sentence.")
    print(result.text)


asyncio.run(main())
```

## Next steps

If you want to...

- add capabilities to your agent, read the tools guide
- build manager-style or handoff-based systems, read the multi-agent guides
- stream events in more detail, use `stream_invoke(...)` / `a_stream_invoke(...)`
- inspect orchestration metadata, use `AgentRunner`
