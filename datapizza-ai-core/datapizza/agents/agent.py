import inspect
from collections.abc import AsyncGenerator, Callable, Generator
from functools import wraps
from threading import Lock
from typing import Any, Literal, Union, cast

from pydantic import BaseModel

from datapizza.agents.logger import AgentLogger
from datapizza.core.clients import Client, ClientResponse
from datapizza.core.clients.models import TokenUsage
from datapizza.core.executors.async_executor import AsyncExecutor
from datapizza.core.utils import sum_token_usage
from datapizza.memory import Memory
from datapizza.tools import Tool
from datapizza.tracing.tracing import agent_span, tool_span
from datapizza.type import (
    ROLE,
    Block,
    FunctionCallBlock,
    FunctionCallResultBlock,
    TextBlock,
)

PLANNING_PROMT = """in this moment you just tell me what you are going to do.
You need to define the next steps to solve the task.
Do not use tools to solve the task.
Do not solve the task, just plan the next steps.
"""


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


class Plan(BaseModel):
    task: str
    steps: list[str]

    def __str__(self):
        separator = "\n - "
        return f"I need to solve the task:\n\n{self.task}\n\nHere is the plan:\n\n - {separator.join(self.steps)}"


class Agent:
    name: str
    system_prompt: str = "You are a helpful assistant."

    def __init__(
        self,
        name: str | None = None,
        client: Client | None = None,
        *,
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
        planning_prompt: str = PLANNING_PROMT,
    ):
        """
        Initialize the agent.

        Args:
            name (str, optional): The name of the agent. Defaults to None.
            client (Client): The client to use for the agent. Defaults to None.
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

        self.system_prompt = system_prompt or self.system_prompt
        if not isinstance(self.system_prompt, str):
            raise ValueError("System prompt must be a string")

        self._client = client
        self._tools = tools or []
        self._planning_interval = planning_interval
        self._planning_prompt = planning_prompt
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

    def can_call(self, agent: Union[list["Agent"], "Agent"]):
        if isinstance(agent, Agent):
            agent = [agent]

        for a in agent:
            self._tools.append(a.as_tool())

    @classmethod
    def _tool_from_agent(cls, agent: "Agent"):
        async def invoke_agent(input_task: str):
            return cast(StepResult, await agent.a_run(input_task)).text

        a_tool = Tool(
            func=invoke_agent,
            name=agent.name,
            description=agent.__doc__,
        )
        return a_tool

    @staticmethod
    def _lock_if_not_stateless(func: Callable):
        @wraps(func)
        def decorated(self, *args, **kwargs):
            if not self._stateless and inspect.isgeneratorfunction(func):
                # For generators, we need a locking wrapper
                def locking_generator():
                    with self._lock:
                        yield from func(self, *args, **kwargs)

                return locking_generator()
            elif not self._stateless:
                with self._lock:
                    return func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return decorated

    def as_tool(self):
        return Agent._tool_from_agent(self)

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

    @_lock_if_not_stateless
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
        yield from self._invoke_stream(task_input, tool_choice, **gen_kwargs)

    @_lock_if_not_stateless
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
        async for step in self._a_invoke_stream(task_input, tool_choice, **gen_kwargs):
            yield step

    def _invoke_stream(
        self, task_input: str, tool_choice, **kwargs
    ) -> Generator[ClientResponse | StepResult | Plan | None, None]:
        self._logger.debug("STARTING AGENT")
        final_answer = None
        current_steps = 1
        memory = self._memory.copy()
        original_task = task_input

        while final_answer is None and (
            self._max_steps is None
            or (self._max_steps and current_steps <= self._max_steps)
        ):
            kwargs["tool_choice"] = tool_choice
            if tool_choice == "required_first":
                if current_steps == 1:
                    kwargs["tool_choice"] = "required"
                else:
                    kwargs["tool_choice"] = "auto"

            self._logger.debug(f"--- STEP {current_steps} ---")

            # Planning step if interval is set
            if self._planning_interval and (
                current_steps == 1 or (current_steps - 1) % self._planning_interval == 0
            ):
                plan = self._create_planning_prompt(
                    original_task, memory, current_steps
                )
                assert isinstance(plan, Plan)
                memory.add_turn(
                    TextBlock(content=str(plan)),
                    role=ROLE.ASSISTANT,
                )
                memory.add_turn(
                    TextBlock(content="Ok, go ahead and now execute the plan."),
                    role=ROLE.USER,
                )

                yield plan

                self._logger.log_panel(str(plan), title="PLAN")

            # Execute planning step
            step_output = None
            for result in self._execute_planning_step(
                current_steps, original_task, memory, **kwargs
            ):
                if isinstance(result, ClientResponse):
                    yield result
                elif isinstance(result, StepResult):
                    step_output = result.text
                    yield result

            if step_output and self._terminate_on_text:
                final_answer = step_output
                break

            current_steps += 1
            original_task = ""

        # Yield final answer if we have one
        if final_answer:
            self._logger.log_panel(final_answer, title="FINAL ANSWER")

        if not self._stateless:
            self._memory = memory

    async def _a_invoke_stream(
        self, task_input: str, tool_choice, **kwargs
    ) -> AsyncGenerator[ClientResponse | StepResult | Plan | None]:
        self._logger.debug("STARTING AGENT")
        final_answer = None
        current_steps = 1
        memory = self._memory.copy()
        original_task = task_input

        while final_answer is None and (
            self._max_steps is None
            or (self._max_steps and current_steps <= self._max_steps)
        ):
            kwargs["tool_choice"] = tool_choice
            if tool_choice == "required_first":
                if current_steps == 1:
                    kwargs["tool_choice"] = "required"
                else:
                    kwargs["tool_choice"] = "auto"

            # step_action = StepResult(index=current_steps)
            self._logger.debug(f"--- STEP {current_steps} ---")
            # yield step_action

            # Planning step if interval is set
            if self._planning_interval and (
                current_steps == 1 or (current_steps - 1) % self._planning_interval == 0
            ):
                plan = await self._a_create_planning_prompt(
                    original_task, memory, current_steps
                )
                assert isinstance(plan, Plan)
                memory.add_turn(
                    TextBlock(content=str(plan)),
                    role=ROLE.ASSISTANT,
                )
                memory.add_turn(
                    TextBlock(content="Ok, go ahead and now execute the plan."),
                    role=ROLE.USER,
                )

                yield plan

                self._logger.log_panel(str(plan), title="PLAN")

            # Execute planning step
            step_output = None
            async for result in self._a_execute_planning_step(
                current_steps, original_task, memory, **kwargs
            ):
                if isinstance(result, ClientResponse):
                    yield result
                elif isinstance(result, StepResult):
                    step_output = result.text
                    yield result

            if step_output and self._terminate_on_text:
                final_answer = step_output
                break

            # task_input = None
            current_steps += 1
            original_task = ""

        # Yield final answer if we have one
        if final_answer:
            self._logger.log_panel(final_answer, title="FINAL ANSWER")

        if not self._stateless:
            self._memory = memory

    def _create_planning_prompt(
        self, original_task: str, memory: Memory, step_number: int
    ) -> Plan:
        """Create a planning prompt that asks the agent to define next steps."""

        prompt = self.system_prompt + self._planning_prompt

        client_response = self._client.structured_response(
            input=original_task,
            tools=self._tools,
            tool_choice="none",
            memory=memory,
            system_prompt=prompt,
            output_cls=Plan,
        )
        return Plan(**client_response.structured_data[0].model_dump())

    async def _a_create_planning_prompt(
        self, original_task: str, memory: Memory, step_number: int
    ) -> Plan:
        """Create a planning prompt that asks the agent to define next steps."""
        prompt = self.system_prompt + self._planning_prompt

        client_response = await self._client.a_structured_response(
            input=original_task,
            tools=self._tools,
            tool_choice="none",
            memory=memory,
            system_prompt=prompt,
            output_cls=Plan,
        )
        return Plan(**client_response.structured_data[0].model_dump())

    def _execute_planning_step(
        self, current_step, planning_prompt: str, memory: Memory, **kwargs
    ) -> Generator[StepResult | ClientResponse, None, None]:
        """Execute a planning step with streaming support."""
        tool_results = []
        step_usage = TokenUsage()

        # Check if streaming is enabled
        response: ClientResponse
        if self._stream:
            for chunk in self._client.stream_invoke(
                input=planning_prompt,
                tools=self._tools,
                memory=memory,
                system_prompt=self.system_prompt,
                **kwargs,
            ):
                step_usage += chunk.usage
                response = chunk
                if chunk.delta:
                    yield chunk

        else:
            # Use regular non-streaming generation
            response = self._client.invoke(
                input=planning_prompt,
                tools=self._tools,
                memory=memory,
                system_prompt=self.system_prompt,
                **kwargs,
            )
            step_usage += response.usage

        if not response:
            raise RuntimeError("No response from client")

        if planning_prompt:
            memory.add_turn(TextBlock(content=planning_prompt), role=ROLE.USER)

        if response and response.text:
            memory.add_turn(TextBlock(content=response.text), role=ROLE.ASSISTANT)

        if response and response.function_calls:
            memory.add_turn(response.function_calls, role=ROLE.ASSISTANT)

        for tool_call in response.function_calls:
            tool_results.append(self._execute_tool(tool_call))

        if tool_results:
            for x in tool_results:
                memory.add_turn(x, role=ROLE.TOOL)

        step_action = StepResult(
            index=current_step,
            content=response.content + tool_results,
            usage=response.usage,
        )

        yield step_action

    async def _a_execute_planning_step(
        self, current_step, planning_prompt: str, memory: Memory, **kwargs
    ) -> AsyncGenerator[StepResult | ClientResponse, None]:
        """Execute a planning step with streaming support."""
        tool_results = []
        step_usage = TokenUsage()
        # Check if streaming is enabled
        response: ClientResponse
        if self._stream:
            async for chunk in self._client.a_stream_invoke(
                input=planning_prompt,
                tools=self._tools,
                memory=memory,
                system_prompt=self.system_prompt,
                **kwargs,
            ):
                step_usage += chunk.usage
                response = chunk
                if chunk.delta:
                    yield chunk

        else:
            # Use regular non-streaming generation
            response = await self._client.a_invoke(
                input=planning_prompt,
                tools=self._tools,
                memory=memory,
                system_prompt=self.system_prompt,
                **kwargs,
            )
            step_usage += response.usage

        if planning_prompt:
            memory.add_turn(TextBlock(content=planning_prompt), role=ROLE.USER)

        if response.text:
            memory.add_turn(TextBlock(content=response.text), role=ROLE.ASSISTANT)

        if response.function_calls:
            memory.add_turn(response.function_calls, role=ROLE.ASSISTANT)

        for tool_call in response.function_calls:
            tool_results.append(await self._a_execute_tool(tool_call))

        if tool_results:
            for x in tool_results:
                memory.add_turn(x, role=ROLE.TOOL)

        step_action = StepResult(
            index=current_step,
            content=response.content + tool_results,
            usage=response.usage,
        )

        yield step_action

    def _execute_tool(
        self, function_call: FunctionCallBlock
    ) -> FunctionCallResultBlock:
        with tool_span(f"Tool {function_call.tool.name}"):
            result = function_call.tool(**function_call.arguments)

            if inspect.iscoroutine(result):
                result = AsyncExecutor.get_instance().run(result)

            if result:
                self._logger.log_panel(
                    result,
                    title=f"TOOL {function_call.tool.name.upper()} RESULT",
                    subtitle="args: " + str(function_call.arguments),
                )
            return FunctionCallResultBlock(
                id=function_call.id,
                tool=function_call.tool,
                result=result,
            )

    async def _a_execute_tool(
        self, function_call: FunctionCallBlock
    ) -> FunctionCallResultBlock:
        with tool_span(f"Tool {function_call.tool.name}"):
            result = function_call.tool(**function_call.arguments)

            if inspect.iscoroutine(result):
                result = await result

            if result:
                self._logger.log_panel(
                    result,
                    title=f"TOOL {function_call.tool.name.upper()} RESULT",
                    subtitle="args: " + str(function_call.arguments),
                )
            return FunctionCallResultBlock(
                id=function_call.id,
                tool=function_call.tool,
                result=result,
            )

    @_lock_if_not_stateless
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
        with agent_span(f"Agent {self.name}"):
            usage = TokenUsage()
            steps = list[ClientResponse | StepResult | Plan | None](
                self._invoke_stream(task_input, tool_choice, **gen_kwargs)
            )
            usage += sum_token_usage(
                [step.usage for step in steps if isinstance(step, StepResult)]
            )

            last_step = cast(
                StepResult,
                steps[-1],
            )
            last_step.usage = usage
            return last_step

    @_lock_if_not_stateless
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
        with agent_span(f"Agent {self.name}"):
            total_usage = TokenUsage()
            results = []
            async for result in self._a_invoke_stream(
                task_input, tool_choice, **gen_kwargs
            ):
                results.append(result)

            total_usage += sum_token_usage(
                [result.usage for result in results if isinstance(result, StepResult)]
            )
            last_result = results[-1] if results else None
            if last_result:
                last_result.usage = total_usage
            return last_result
