import asyncio
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, Union, cast

from pydantic import BaseModel

from datapizza.agents.logger import AgentLogger
from datapizza.core.clients import Client, ClientResponse
from datapizza.core.clients.models import TokenUsage
from datapizza.memory import Memory
from datapizza.tools import Tool
from datapizza.type import (
    Block,
    FunctionCallBlock,
    StructuredBlock,
    TextBlock,
)

PLANNING_PROMPT = """in this moment you just tell me what you are going to do.
You need to define the next steps to solve the task.
Do not use tools to solve the task.
Do not solve the task, just plan the next steps.
"""

PLANNING_PROMT = PLANNING_PROMPT


class StepResult:
    def __init__(
        self,
        index: int,
        content: list[Block],
        usage: TokenUsage | None = None,
    ):
        self.index = index
        self.content = content
        self.usage = usage or TokenUsage()

    @property
    def text(self) -> str:
        return "\n".join(
            block.content for block in self.content if isinstance(block, TextBlock)
        )

    @property
    def tools_used(self) -> list[FunctionCallBlock]:
        return [block for block in self.content if isinstance(block, FunctionCallBlock)]

    @property
    def structured_data(self) -> list[BaseModel]:
        return [
            block.content
            for block in self.content
            if isinstance(block, StructuredBlock)
        ]


class Plan(BaseModel):
    task: str
    steps: list[str]

    def __str__(self):
        separator = "\n - "
        return f"I need to solve the task:\n\n{self.task}\n\nHere is the plan:\n\n - {separator.join(self.steps)}"


@dataclass
class StepContext:
    agent: "Agent"
    step_index: int
    task_input: str
    memory: Memory
    tool_choice: Literal["auto", "required", "none", "required_first"] | list[str]


class AgentHooks:
    def before_step(self, context: StepContext) -> None:
        pass

    def after_step(self, context: StepContext, result: StepResult) -> None:
        pass


class Agent:
    name: str
    system_prompt: str = "You are a helpful assistant."

    def __init__(
        self,
        name: str | None = None,
        client: Client | None = None,
        *,
        description: str | None = None,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        max_steps: int | None = None,
        terminate_on_text: bool | None = True,
        stateless: bool = True,
        gen_args: dict[str, Any] | None = None,
        memory: Memory | None = None,
        stream: bool | None = None,
        # action_on_stop_reason: dict[str, Action] | None = None,
        can_call: list["Agent"] | None = None,
        logger: AgentLogger | None = None,
        planning_interval: int = 0,
        planning_prompt: str = PLANNING_PROMPT,
        output_cls: type[BaseModel] | None = None,
        hooks: AgentHooks | None = None,
        handoffs: list["Agent"] | None = None,
    ):
        """
        Initialize the agent.

        Args:
            name (str, optional): The name of the agent. Defaults to None.
            client (Client): The client to use for the agent. Defaults to None.
            description (str, optional): Human-readable description used when converting the agent to a tool. Defaults to None.
            system_prompt (str, optional): The system prompt to use for the agent. Defaults to None.

            tools (list[Tool], optional): A list of tools to use with the agent. Defaults to None.
            max_steps (int, optional): The maximum number of steps to execute. Defaults to None.
            terminate_on_text (bool, optional): Whether to terminate the agent on text. Defaults to True.
            stateless (bool, optional): Whether to use stateless execution. Defaults to True.
            gen_args (dict[str, Any], optional): Additional arguments to pass to the agent's execution. Defaults to None.
            memory (Memory, optional): The memory to use for the agent. Defaults to None.
            stream (bool, optional): Whether to stream the agent's execution. Defaults to None.
            can_call (list[Agent], optional): A list of agents that can call the agent. Defaults to None.
            logger (AgentLogger, optional): The logger to use for the agent. Defaults to None.
            planning_interval (int, optional): The planning interval to use for the agent. Defaults to 0.
            planning_prompt (str, optional): The planning prompt to use for the agent planning steps. Defaults to PLANNING_PROMT.
            output_cls (type[BaseModel], optional): If set, every agent model turn requests structured output of this class.
            hooks (AgentHooks, optional): Hook callbacks invoked before and after each step.

        """
        if not client:
            raise ValueError("Client is required")

        if not name and not getattr(self, "name", None):
            raise ValueError(
                "Name is required, you can pass it as a parameter or set it in the agent class"
            )

        if not system_prompt and not getattr(self, "system_prompt", None):
            raise ValueError(
                "System prompt is required, you can pass it as a parameter or set it in the agent class"
            )

        self.name = name or self.name
        if not isinstance(self.name, str):
            raise ValueError("Name must be a string")

        self.description = description
        if self.description is not None and not isinstance(self.description, str):
            raise ValueError("Description must be a string")

        self.system_prompt = system_prompt or self.system_prompt
        if not isinstance(self.system_prompt, str):
            raise ValueError("System prompt must be a string")

        self._client = client
        self._tools = tools or []
        self._handoffs = handoffs or []
        self._planning_interval = planning_interval
        self._planning_prompt = planning_prompt
        self._output_cls = output_cls
        self._hooks = hooks
        self._memory = memory or Memory()
        self._stateless = stateless

        if can_call:
            self.can_call(can_call)

        self._max_steps = max_steps
        self._terminate_on_text = terminate_on_text
        self._stream = stream

        if not logger:
            self._logger = AgentLogger(agent_name=self.name)
        else:
            self._logger = logger

        for tool in self._decorator_tools():
            self._add_tool(tool)

        self._lock = Lock()
        self._async_lock = asyncio.Lock()

    def can_call(self, agent: Union[list["Agent"], "Agent"]):
        if isinstance(agent, Agent):
            agent = [agent]

        for a in agent:
            self._tools.append(a.as_tool())

    def can_handoff(self, agent: Union[list["Agent"], "Agent"]):
        if isinstance(agent, Agent):
            agent = [agent]

        self._handoffs.extend(agent)

    @classmethod
    def _tool_from_agent(cls, agent: "Agent", end: bool = False):
        async def invoke_agent(input_task: str):
            result = await agent.a_run(input_task)
            step_result = cast(StepResult, result)
            if step_result.structured_data:
                return "\n".join(
                    item.model_dump_json() for item in step_result.structured_data
                )
            return step_result.text

        tool_description = (
            getattr(agent, "description", None) or agent.__doc__ or agent.name
        )

        a_tool = Tool(
            func=invoke_agent,
            name=agent.name,
            description=tool_description,
            end=end,
        )
        return a_tool

    @staticmethod
    def _contains_ending_tool(step: StepResult) -> bool:
        content = step.content
        return any(
            block.tool.end_invoke
            for block in content
            if isinstance(block, FunctionCallBlock)
        )

    def as_tool(self, end: bool = False):
        return Agent._tool_from_agent(self, end=end)

    def _add_tool(self, tool: Tool):
        self._tools.append(tool)

    def _decorator_tools(self):
        tools = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            # Check for tool methods
            if isinstance(attr, Tool):
                tools.append(attr)

        return tools

    def stream_invoke(
        self,
        task_input: str,
        tool_choice: Literal["auto", "required", "none", "required_first"]
        | list[str] = "auto",
        **gen_kwargs,
    ) -> Generator[ClientResponse | StepResult | Plan | None, None]:
        """
        Stream the agent's execution, yielding intermediate steps and final result.

        Args:
            task_input (str): The input text/prompt to send to the model
            tool_choice (Literal["auto", "required", "none", "required_first"] | list[str], optional): Controls which tool to use ("auto" by default)
            **gen_kwargs: Additional keyword arguments to pass to the agent's execution

        Yields:
            The intermediate steps and final result of the agent's execution

        """
        from datapizza.agents.runner import AgentRunner

        yield from AgentRunner().stream(self, task_input, tool_choice, **gen_kwargs)

    async def a_stream_invoke(
        self,
        task_input: str,
        tool_choice: Literal["auto", "required", "none", "required_first"]
        | list[str] = "auto",
        **gen_kwargs,
    ) -> AsyncGenerator[ClientResponse | StepResult | Plan | None]:
        """
        Stream the agent's execution asynchronously, yielding intermediate steps and final result.

        Args:
            task_input (str): The input text/prompt to send to the model
            tool_choice (Literal["auto", "required", "none", "required_first"] | list[str], optional): Controls which tool to use ("auto" by default)
            **gen_kwargs: Additional keyword arguments to pass to the agent's execution

        Yields:
            The intermediate steps and final result of the agent's execution

        """
        from datapizza.agents.runner import AgentRunner

        async for step in AgentRunner().a_stream(
            self, task_input, tool_choice, **gen_kwargs
        ):
            yield step

    def run(
        self,
        task_input: str,
        tool_choice: Literal["auto", "required", "none", "required_first"]
        | list[str] = "auto",
        **gen_kwargs,
    ) -> StepResult | None:
        """
        Run the agent on a task input.

        Args:
            task_input (str): The input text/prompt to send to the model
            tool_choice (Literal["auto", "required", "none", "required_first"] | list[str], optional): Controls which tool to use ("auto" by default)
            **gen_kwargs: Additional keyword arguments to pass to the agent's execution

        Returns:
            The final result of the agent's execution
        """
        from datapizza.agents.runner import AgentRunner

        result = AgentRunner().run(self, task_input, tool_choice, **gen_kwargs)
        return result.final_step

    async def a_run(
        self,
        task_input: str,
        tool_choice: Literal["auto", "required", "none", "required_first"]
        | list[str] = "auto",
        **gen_kwargs,
    ) -> StepResult | None:
        """
        Run the agent on a task input asynchronously.

        Args:
            task_input (str): The input text/prompt to send to the model
            tool_choice (Literal["auto", "required", "none", "required_first"] | list[str], optional): Controls which tool to use ("auto" by default)
            **gen_kwargs: Additional keyword arguments to pass to the agent's execution

        Returns:
            The final result of the agent's execution
        """
        from datapizza.agents.runner import AgentRunner

        result = await AgentRunner().a_run(self, task_input, tool_choice, **gen_kwargs)
        return result.final_step
