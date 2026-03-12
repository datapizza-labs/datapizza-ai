import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import BaseModel

from datapizza.agents.agent import (
    PLANNING_PROMT,
    Agent,
    AgentHooks,
    StepContext,
    StepResult,
)
from datapizza.agents.runner import AgentRunner
from datapizza.clients import MockClient
from datapizza.core.clients import ClientResponse
from datapizza.tools import tool
from datapizza.type import FunctionCallBlock, TextBlock


class TestBaseAgents:
    def test_agent_defaults(self):
        agent = Agent(name="datapizza_agent", client=MockClient())
        assert agent.name == "datapizza_agent"
        assert agent.system_prompt == "You are a helpful assistant."
        assert agent._planning_prompt == PLANNING_PROMT

    def test_init_agent(self):
        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            planning_prompt="test planning prompt",
        )
        assert agent.name == "test"
        assert agent._planning_prompt == "test planning prompt"

    def test_invoke_agent(self):
        agent = Agent(
            name="test", client=MockClient(), system_prompt="You are a test agent"
        )
        assert agent.run("Hello").text == "Hello"

    def test_a_invoke_agent(self):
        agent = Agent(
            name="test", client=MockClient(), system_prompt="You are a test agent"
        )
        res = asyncio.run(agent.a_run("Hello"))  # type: ignore
        assert res.text == "Hello"

    def test_stream_invoke_agent(self):
        agent = Agent(
            name="test", client=MockClient(), system_prompt="You are a test agent"
        )
        res = list(agent.stream_invoke("Hello"))
        assert isinstance(res[0], StepResult)
        assert res[0].index == 1
        assert res[0].text == "Hello"

    def test_can_call_agent(self):
        agent1 = Agent(
            name="test1", client=MockClient(), system_prompt="You are a test agent"
        )
        agent2 = Agent(
            name="test2", client=MockClient(), system_prompt="You are a test agent"
        )
        agent1.can_call(agent2)
        assert agent1._tools[0].name == agent2.as_tool().name
        assert agent1._tools[0].description == agent2.as_tool().description

        agent_aggregator = Agent(
            name="test_aggregator",
            client=MockClient(),
            system_prompt="You are a test agent",
            can_call=[agent1, agent2],
        )
        assert agent_aggregator._tools[0].name == agent1.as_tool().name
        assert agent_aggregator._tools[1].name == agent2.as_tool().name

    def test_as_tool_end_invoke_true(self):
        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
        )

        a_tool = agent.as_tool(end=True)
        assert a_tool.end_invoke is True

    def test_as_tool_end_invoke_default_false(self):
        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
        )

        a_tool = agent.as_tool()
        assert a_tool.end_invoke is False

    def test_as_tool_prefers_instance_description(self):
        agent = Agent(
            name="test",
            client=MockClient(),
            description="My custom tool description",
            system_prompt="You are a test agent",
        )

        a_tool = agent.as_tool()
        assert a_tool.description == "My custom tool description"

    def test_as_tool_description_fallback_to_docstring(self):
        class TestDocAgent(Agent):
            """Tool description from class docstring."""

            name = "doc_agent"
            system_prompt = "You are a test agent"

        agent = TestDocAgent(client=MockClient())
        a_tool = agent.as_tool()
        assert a_tool.description == "Tool description from class docstring."

    def test_as_tool_description_fallback_to_name(self):
        agent = Agent(
            name="name_fallback_agent",
            client=MockClient(),
            system_prompt="You are a test agent",
        )

        a_tool = agent.as_tool()
        assert a_tool.description == "name_fallback_agent"

    def test_as_tool_empty_description_fallback_to_name(self):
        agent = Agent(
            name="empty_description_agent",
            client=MockClient(),
            description="",
            system_prompt="You are a test agent",
        )

        a_tool = agent.as_tool()
        assert a_tool.description == "empty_description_agent"

    def test_params_as_class_attributes(self):
        class TestAgent(Agent):
            name = "test"
            system_prompt = "You are a test agent"

        client = MockClient()
        agent = TestAgent(client=client)
        assert agent.name == "test"
        assert agent._client == client
        assert agent.system_prompt == "You are a test agent"

    def test_tools_as_class_attributes(self):
        class TestAgent(Agent):
            name = "test"
            system_prompt = "You are a test agent"

            @tool
            def test_tool(self, x: str) -> str:
                return x

        agent = TestAgent(client=MockClient())
        assert agent._tools[0].name == "test_tool"

    def test_agent_stream_text(self):
        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            stream=True,
        )
        res = list(agent.stream_invoke("Hello, how are you?"))
        assert isinstance(res[0], ClientResponse)
        assert res[0].text == "H"
        assert res[1].text == "He"
        assert res[2].text == "Hel"

    def test_agent_structured_output_sync(self):
        class Person(BaseModel):
            name: str
            age: int

        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            output_cls=Person,
        )

        res = agent.run('{"name": "Alice", "age": 30}')
        assert res
        assert res.index == 1
        assert res.text == ""
        assert len(res.structured_data) == 1
        assert res.structured_data[0] == Person(name="Alice", age=30)

    def test_agent_structured_output_hard_fail_on_unsupported_client(self):
        class Person(BaseModel):
            name: str

        class UnsupportedStructuredMockClient(MockClient):
            def _structured_response(self, *args, **kwargs):
                raise NotImplementedError("not supported")

        agent = Agent(
            name="test",
            client=UnsupportedStructuredMockClient(),
            system_prompt="You are a test agent",
            output_cls=Person,
        )

        with pytest.raises(ValueError, match="does not support structured responses"):
            agent.run('{"name": "Alice"}')

    def test_agent_structured_output_async_hard_fail_on_unsupported_client(self):
        class Person(BaseModel):
            name: str

        class UnsupportedStructuredMockClient(MockClient):
            def _a_structured_response(self, *args, **kwargs):
                raise NotImplementedError("not supported")

        agent = Agent(
            name="test",
            client=UnsupportedStructuredMockClient(),
            system_prompt="You are a test agent",
            output_cls=Person,
        )

        with pytest.raises(
            ValueError, match="does not support async structured responses"
        ):
            asyncio.run(agent.a_run('{"name": "Alice"}'))

    def test_agent_hooks_run_single_step(self):
        events = []

        class TestHooks(AgentHooks):
            def before_step(self, context: StepContext) -> None:
                events.append(("before", context.step_index, context.task_input))

            def after_step(self, context: StepContext, result: StepResult) -> None:
                events.append(("after", context.step_index, result.text))

        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            hooks=TestHooks(),
        )

        res = agent.run("Hello")

        assert res.text == "Hello"
        assert events == [("before", 1, "Hello"), ("after", 1, "Hello")]

    def test_agent_hooks_run_multiple_steps(self):
        events = []

        class TestHooks(AgentHooks):
            def before_step(self, context: StepContext) -> None:
                events.append(("before", context.step_index, context.task_input))

            def after_step(self, context: StepContext, result: StepResult) -> None:
                events.append(("after", context.step_index, result.text))

        @tool(end=False)
        def test_tool(*args, **kwargs):
            return "tool called"

        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            tools=[test_tool],
            hooks=TestHooks(),
        )

        res = agent.run("function call")

        assert res.index == 2
        assert events == [
            ("before", 1, "function call"),
            ("after", 1, ""),
            ("before", 2, ""),
            ("after", 2, "tool called"),
        ]

    def test_agent_hooks_async(self):
        events = []

        class TestHooks(AgentHooks):
            def before_step(self, context: StepContext) -> None:
                events.append(("before", context.step_index, context.task_input))

            def after_step(self, context: StepContext, result: StepResult) -> None:
                events.append(("after", context.step_index, result.text))

        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            hooks=TestHooks(),
        )

        res = asyncio.run(agent.a_run("Hello"))

        assert res.text == "Hello"
        assert events == [("before", 1, "Hello"), ("after", 1, "Hello")]

    def test_agent_hooks_exception_bubbles(self):
        class TestHooks(AgentHooks):
            def before_step(self, context: StepContext) -> None:
                raise RuntimeError("hook failed")

        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            hooks=TestHooks(),
        )

        with pytest.raises(RuntimeError, match="hook failed"):
            agent.run("Hello")


class TestStatelessAgents:
    def test_stateless_agent_invoke(self):
        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            stateless=True,
        )
        assert agent.run("Hello").text == "Hello"
        assert len(agent._memory) == 0

    def test_stateless_agent_stream_invoke(self):
        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            stateless=True,
        )
        res = list(agent.stream_invoke("Hello, how are you?"))
        assert isinstance(res[0], StepResult)
        assert res[0].text == "Hello, how are you?"
        assert len(agent._memory) == 0

    def test_not_stateless_agent_invoke(self):
        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            stateless=False,
        )
        assert agent.run("Hello").text == "Hello"
        assert len(agent._memory) == 2

        agent.run("Hello")
        assert len(agent._memory) == 4

    def test_non_stateless_lock_async(self):
        async def test_func():
            agent = Agent(
                name="test",
                client=MockClient(),
                system_prompt="You are a test agent",
                stateless=False,
            )

            tasks = []
            for x in range(3):
                tasks.append(asyncio.create_task(agent.a_run(str(x))))

            assert len(agent._memory) == 0

            res = await asyncio.gather(*tasks)

            assert len(agent._memory) == 6
            return res

        asyncio.run(test_func())

    def test_non_stateless_lock_thread(self):
        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            stateless=False,
        )

        def test_func(x):
            agent.run(str(x))

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(test_func, x) for x in range(9)]
            [future.result() for future in futures]

        assert len(agent._memory) == 18

    def test_non_stateless_stream_invoke_lock(self):
        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            stateless=False,
        )
        res = agent.stream_invoke("Hello, how are you?")
        res2 = agent.stream_invoke("Hello, how are you?")
        assert isinstance(next(res), StepResult)
        list(res)
        assert isinstance(next(res2), StepResult)
        list(res2)

        assert len(agent._memory) == 4

    def test_stateless_stream_invoke_lock(self):
        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            stateless=True,
        )
        res = agent.stream_invoke("Hello, how are you?")
        res2 = agent.stream_invoke("Hello, how are you?")

        next(res)
        next(res2)

        list(res)
        list(res2)

        assert len(agent._memory) == 0

    def test_tools_with_end_invoke(self):
        @tool(end=True)
        def test_tool(*args, **kwargs):
            return "tool called"

        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            tools=[test_tool],
        )

        res = agent.run("function call")
        assert len(res.tools_used)
        assert res.index == 1

    def test_tools_with_not_end_invoke(self):
        @tool(end=False)
        def test_tool(*args, **kwargs):
            return "tool called"

        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            tools=[test_tool],
        )

        res = agent.run("function call")
        assert res.index == 2

    def test_terminate_on_text_ignores_text_when_tool_call_is_present(self):
        @tool(end=False)
        def test_tool(*args, **kwargs):
            return "tool called"

        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            tools=[test_tool],
            terminate_on_text=True,
        )

        res = agent.run("mixed function call")
        assert res.index == 2
        assert res.text == "tool called"

    def test_a_terminate_on_text_ignores_text_when_tool_call_is_present(self):
        @tool(end=False)
        def test_tool(*args, **kwargs):
            return "tool called"

        agent = Agent(
            name="test",
            client=MockClient(),
            system_prompt="You are a test agent",
            tools=[test_tool],
            terminate_on_text=True,
        )

        res = asyncio.run(agent.a_run("mixed function call"))
        assert res.index == 2
        assert res.text == "tool called"


class HandoffMockClient(MockClient):
    def _invoke(self, input, tools=None, memory=None, **kwargs):
        if tools is None:
            tools = []

        input_text = ""
        if isinstance(input, list):
            for block in input:
                if isinstance(block, TextBlock):
                    input_text = block.content

        if input_text == "handoff":
            handoff_tool = next(
                tool for tool in tools if tool.name == "transfer_to_specialist"
            )
            return ClientResponse(
                content=[
                    FunctionCallBlock(
                        id="handoff-1",
                        arguments={
                            "task_input": "delegated",
                            "reason": "needs specialist",
                        },
                        name=handoff_tool.name,
                        tool=handoff_tool,
                    )
                ]
            )

        if input_text == "delegated":
            return ClientResponse(
                content=[TextBlock(content=f"shared:{len(memory) if memory else 0}")]
            )

        return super()._invoke(input=input, tools=tools, memory=memory, **kwargs)

    async def _a_invoke(self, input, tools=None, memory=None, **kwargs):
        return self._invoke(input=input, tools=tools, memory=memory, **kwargs)


class TestAgentRunner:
    def test_agent_run_preserves_stepresult_with_handoff(self):
        specialist = Agent(
            name="specialist",
            client=HandoffMockClient(),
            system_prompt="You are a specialist",
        )
        root = Agent(
            name="root",
            client=HandoffMockClient(),
            system_prompt="You are a root agent",
            stateless=False,
            handoffs=[specialist],
        )

        result = root.run("handoff")

        assert isinstance(result, StepResult)
        assert result.text == "shared:3"
        assert len(root._memory) == 5

    def test_runner_returns_high_level_handoff_result(self):
        specialist = Agent(
            name="specialist",
            client=HandoffMockClient(),
            system_prompt="You are a specialist",
        )
        root = Agent(
            name="root",
            client=HandoffMockClient(),
            system_prompt="You are a root agent",
            handoffs=[specialist],
        )

        result = AgentRunner().run(root, "handoff")

        assert result.final_step is not None
        assert result.final_step.text == "shared:3"
        assert result.final_agent is specialist
        assert result.handoff_count == 1
        assert result.visited_agents == ["root", "specialist"]
        memory_dict = result.memory.to_dict()
        serialized_memory = str(memory_dict)
        assert "Transfer to specialist completed" in serialized_memory

    def test_runner_async_handoff(self):
        specialist = Agent(
            name="specialist",
            client=HandoffMockClient(),
            system_prompt="You are a specialist",
        )
        root = Agent(
            name="root",
            client=HandoffMockClient(),
            system_prompt="You are a root agent",
            handoffs=[specialist],
        )

        result = asyncio.run(root.a_run("handoff"))

        assert result is not None
        assert result.text == "shared:3"

    def test_stream_invoke_handoff(self):
        specialist = Agent(
            name="specialist",
            client=HandoffMockClient(),
            system_prompt="You are a specialist",
        )
        root = Agent(
            name="root",
            client=HandoffMockClient(),
            system_prompt="You are a root agent",
            handoffs=[specialist],
        )

        results = list(root.stream_invoke("handoff"))

        assert isinstance(results[-1], StepResult)
        assert results[-1].text == "shared:3"

    def test_a_stream_invoke_handoff(self):
        async def run_stream():
            specialist = Agent(
                name="specialist",
                client=HandoffMockClient(),
                system_prompt="You are a specialist",
            )
            root = Agent(
                name="root",
                client=HandoffMockClient(),
                system_prompt="You are a root agent",
                handoffs=[specialist],
            )

            results = []
            async for item in root.a_stream_invoke("handoff"):
                results.append(item)
            return results

        results = asyncio.run(run_stream())

        assert isinstance(results[-1], StepResult)
        assert results[-1].text == "shared:3"
