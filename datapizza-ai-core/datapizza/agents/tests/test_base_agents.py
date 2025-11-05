import asyncio
from concurrent.futures import ThreadPoolExecutor

from datapizza.agents.agent import PLANNING_PROMT, Agent, StepResult
from datapizza.clients import MockClient
from datapizza.core.clients import ClientResponse
from datapizza.tools import tool


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
