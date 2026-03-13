import inspect
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from datapizza.core.clients import ClientResponse
from datapizza.core.clients.models import TokenUsage
from datapizza.core.executors.async_executor import AsyncExecutor
from datapizza.core.utils import sum_token_usage
from datapizza.memory import Memory
from datapizza.tools import Tool
from datapizza.tracing.tracing import agent_span, tool_span
from datapizza.type import (
    ROLE,
    FunctionCallBlock,
    FunctionCallResultBlock,
    TextBlock,
)

if TYPE_CHECKING:
    from .agent import Agent, Plan, StepResult


@dataclass
class HandoffRequest:
    target: str
    task_input: str
    reason: str | None = None


@dataclass
class AgentRunnerResult:
    final_step: "StepResult | None"
    final_agent: "Agent | None"
    handoff_count: int
    visited_agents: list[str]
    memory: Memory
    usage: TokenUsage


@dataclass
class _SingleAgentResult:
    final_step: "StepResult | None"
    handoff: HandoffRequest | None
    usage: TokenUsage


class AgentRunner:
    def __init__(self, *, max_handoffs: int = 8):
        self.max_handoffs = max_handoffs

    def run(
        self,
        root_agent: "Agent",
        task_input: str,
        tool_choice: Literal["auto", "required", "none", "required_first"]
        | list[str] = "auto",
        **gen_kwargs,
    ) -> AgentRunnerResult:
        if not root_agent._stateless:
            with root_agent._lock:
                return self._run(root_agent, task_input, tool_choice, **gen_kwargs)
        return self._run(root_agent, task_input, tool_choice, **gen_kwargs)

    async def a_run(
        self,
        root_agent: "Agent",
        task_input: str,
        tool_choice: Literal["auto", "required", "none", "required_first"]
        | list[str] = "auto",
        **gen_kwargs,
    ) -> AgentRunnerResult:
        if not root_agent._stateless:
            async with root_agent._async_lock:
                return await self._a_run(
                    root_agent, task_input, tool_choice, **gen_kwargs
                )
        return await self._a_run(root_agent, task_input, tool_choice, **gen_kwargs)

    def stream(
        self,
        root_agent: "Agent",
        task_input: str,
        tool_choice: Literal["auto", "required", "none", "required_first"]
        | list[str] = "auto",
        **gen_kwargs,
    ) -> "Generator[ClientResponse | StepResult | Plan | None, None, None]":
        if not root_agent._stateless:
            with root_agent._lock:
                yield from self._stream(
                    root_agent, task_input, tool_choice, **gen_kwargs
                )
                return
        yield from self._stream(root_agent, task_input, tool_choice, **gen_kwargs)

    async def a_stream(
        self,
        root_agent: "Agent",
        task_input: str,
        tool_choice: Literal["auto", "required", "none", "required_first"]
        | list[str] = "auto",
        **gen_kwargs,
    ) -> "AsyncGenerator[ClientResponse | StepResult | Plan | None]":
        if not root_agent._stateless:
            async with root_agent._async_lock:
                async for item in self._a_stream(
                    root_agent, task_input, tool_choice, **gen_kwargs
                ):
                    yield item
                return
        async for item in self._a_stream(
            root_agent, task_input, tool_choice, **gen_kwargs
        ):
            yield item

    def _run(
        self,
        root_agent: "Agent",
        task_input: str,
        tool_choice: Literal["auto", "required", "none", "required_first"] | list[str],
        **gen_kwargs,
    ) -> AgentRunnerResult:
        from .agent import StepResult

        shared_memory = self._initial_memory(root_agent)
        current_agent = root_agent
        current_input = task_input
        handoff_count = 0
        visited_agents = [root_agent.name]
        final_step: StepResult | None = None
        total_usage = TokenUsage()

        while True:
            if handoff_count > self.max_handoffs:
                raise RuntimeError("Maximum handoffs exceeded")

            with agent_span(f"Agent {current_agent.name}"):
                result = self._run_single_agent(
                    current_agent,
                    current_input,
                    shared_memory,
                    tool_choice,
                    **gen_kwargs,
                )

            total_usage += result.usage
            final_step = result.final_step

            if not result.handoff:
                break

            handoff_count += 1
            current_agent = self._resolve_handoff(current_agent, result.handoff)
            current_input = result.handoff.task_input
            visited_agents.append(current_agent.name)

        if final_step:
            final_step.usage = total_usage

        if not root_agent._stateless:
            root_agent._memory = shared_memory

        return AgentRunnerResult(
            final_step=final_step,
            final_agent=current_agent,
            handoff_count=handoff_count,
            visited_agents=visited_agents,
            memory=shared_memory,
            usage=total_usage,
        )

    def _stream(
        self,
        root_agent: "Agent",
        task_input: str,
        tool_choice: Literal["auto", "required", "none", "required_first"] | list[str],
        **gen_kwargs,
    ) -> "Generator[ClientResponse | StepResult | Plan | None, None, None]":
        shared_memory = self._initial_memory(root_agent)
        current_agent = root_agent
        current_input = task_input
        handoff_count = 0

        while True:
            if handoff_count > self.max_handoffs:
                raise RuntimeError("Maximum handoffs exceeded")

            handoff: HandoffRequest | None = None
            with agent_span(f"Agent {current_agent.name}"):
                for item in self._stream_single_agent(
                    current_agent,
                    current_input,
                    shared_memory,
                    tool_choice,
                    **gen_kwargs,
                ):
                    if isinstance(item, HandoffRequest):
                        handoff = item
                    else:
                        yield item

            if handoff is None:
                break

            handoff_count += 1
            current_agent = self._resolve_handoff(current_agent, handoff)
            current_input = handoff.task_input

        if not root_agent._stateless:
            root_agent._memory = shared_memory

    async def _a_stream(
        self,
        root_agent: "Agent",
        task_input: str,
        tool_choice: Literal["auto", "required", "none", "required_first"] | list[str],
        **gen_kwargs,
    ) -> "AsyncGenerator[ClientResponse | StepResult | Plan | None]":
        shared_memory = self._initial_memory(root_agent)
        current_agent = root_agent
        current_input = task_input
        handoff_count = 0

        while True:
            if handoff_count > self.max_handoffs:
                raise RuntimeError("Maximum handoffs exceeded")

            handoff: HandoffRequest | None = None
            with agent_span(f"Agent {current_agent.name}"):
                async for item in self._a_stream_single_agent(
                    current_agent,
                    current_input,
                    shared_memory,
                    tool_choice,
                    **gen_kwargs,
                ):
                    if isinstance(item, HandoffRequest):
                        handoff = item
                    else:
                        yield item

            if handoff is None:
                break

            handoff_count += 1
            current_agent = self._resolve_handoff(current_agent, handoff)
            current_input = handoff.task_input

        if not root_agent._stateless:
            root_agent._memory = shared_memory

    async def _a_run(
        self,
        root_agent: "Agent",
        task_input: str,
        tool_choice: Literal["auto", "required", "none", "required_first"] | list[str],
        **gen_kwargs,
    ) -> AgentRunnerResult:
        from .agent import StepResult

        shared_memory = self._initial_memory(root_agent)
        current_agent = root_agent
        current_input = task_input
        handoff_count = 0
        visited_agents = [root_agent.name]
        final_step: StepResult | None = None
        total_usage = TokenUsage()

        while True:
            if handoff_count > self.max_handoffs:
                raise RuntimeError("Maximum handoffs exceeded")

            with agent_span(f"Agent {current_agent.name}"):
                result = await self._a_run_single_agent(
                    current_agent,
                    current_input,
                    shared_memory,
                    tool_choice,
                    **gen_kwargs,
                )

            total_usage += result.usage
            final_step = result.final_step

            if not result.handoff:
                break

            handoff_count += 1
            current_agent = self._resolve_handoff(current_agent, result.handoff)
            current_input = result.handoff.task_input
            visited_agents.append(current_agent.name)

        if final_step:
            final_step.usage = total_usage

        if not root_agent._stateless:
            root_agent._memory = shared_memory

        return AgentRunnerResult(
            final_step=final_step,
            final_agent=current_agent,
            handoff_count=handoff_count,
            visited_agents=visited_agents,
            memory=shared_memory,
            usage=total_usage,
        )

    def _stream_single_agent(
        self,
        agent: "Agent",
        task_input: str,
        memory: Memory,
        tool_choice: Literal["auto", "required", "none", "required_first"] | list[str],
        **kwargs,
    ) -> "Generator[ClientResponse | StepResult | Plan | HandoffRequest | None, None, None]":
        from .agent import Plan, StepContext

        agent._logger.debug("STARTING AGENT")
        final_answer = None
        current_steps = 1
        original_task = task_input

        while final_answer is None and (
            agent._max_steps is None
            or (agent._max_steps and current_steps <= agent._max_steps)
        ):
            kwargs = dict(kwargs)
            kwargs["tool_choice"] = self._resolve_tool_choice(
                tool_choice, current_steps
            )

            context = StepContext(
                agent=agent,
                step_index=current_steps,
                task_input=original_task,
                memory=memory,
                tool_choice=kwargs["tool_choice"],
            )
            if agent._hooks:
                agent._hooks.before_step(context)

            agent._logger.debug(f"--- STEP {current_steps} ---")

            if agent._planning_interval and (
                current_steps == 1
                or (current_steps - 1) % agent._planning_interval == 0
            ):
                plan = self._create_plan(agent, original_task, memory)
                assert isinstance(plan, Plan)
                memory.add_turn(TextBlock(content=str(plan)), role=ROLE.ASSISTANT)
                memory.add_turn(
                    TextBlock(content="Ok, go ahead and now execute the plan."),
                    role=ROLE.USER,
                )
                yield plan
                agent._logger.log_panel(str(plan), title="PLAN")

            response = None
            stream_usage = TokenUsage()
            for chunk in self._invoke_model_stream(
                agent, original_task, memory, **kwargs
            ):
                if isinstance(chunk, ClientResponse):
                    stream_usage += chunk.usage
                    response = chunk
                    if chunk.delta:
                        yield chunk

            if response is None:
                raise RuntimeError("No response from client")
            response.usage = stream_usage

            step_result, handoff = self._finalize_step(
                agent, current_steps, original_task, memory, response
            )
            yield step_result

            if agent._hooks:
                agent._hooks.after_step(context, step_result)

            if handoff is not None:
                yield handoff
                return

            step_output = step_result.text
            step_has_tools = bool(step_result.tools_used)
            step_has_structured = bool(step_result.structured_data)

            if step_output and agent._terminate_on_text and not step_has_tools:
                final_answer = step_output
                break

            if agent._output_cls and step_has_structured and not step_has_tools:
                final_answer = ""
                break

            if agent._contains_ending_tool(step_result):
                agent._logger.debug("ending tool found, ending agent")
                break

            current_steps += 1
            original_task = ""

        if final_answer:
            agent._logger.log_panel(final_answer, title="FINAL ANSWER")

    async def _a_stream_single_agent(
        self,
        agent: "Agent",
        task_input: str,
        memory: Memory,
        tool_choice: Literal["auto", "required", "none", "required_first"] | list[str],
        **kwargs,
    ) -> "AsyncGenerator[ClientResponse | StepResult | Plan | HandoffRequest | None]":
        from .agent import Plan, StepContext

        agent._logger.debug("STARTING AGENT")
        final_answer = None
        current_steps = 1
        original_task = task_input

        while final_answer is None and (
            agent._max_steps is None
            or (agent._max_steps and current_steps <= agent._max_steps)
        ):
            kwargs = dict(kwargs)
            kwargs["tool_choice"] = self._resolve_tool_choice(
                tool_choice, current_steps
            )

            context = StepContext(
                agent=agent,
                step_index=current_steps,
                task_input=original_task,
                memory=memory,
                tool_choice=kwargs["tool_choice"],
            )
            if agent._hooks:
                agent._hooks.before_step(context)

            agent._logger.debug(f"--- STEP {current_steps} ---")

            if agent._planning_interval and (
                current_steps == 1
                or (current_steps - 1) % agent._planning_interval == 0
            ):
                plan = await self._a_create_plan(agent, original_task, memory)
                assert isinstance(plan, Plan)
                memory.add_turn(TextBlock(content=str(plan)), role=ROLE.ASSISTANT)
                memory.add_turn(
                    TextBlock(content="Ok, go ahead and now execute the plan."),
                    role=ROLE.USER,
                )
                yield plan
                agent._logger.log_panel(str(plan), title="PLAN")

            response = None
            stream_usage = TokenUsage()
            async for chunk in self._a_invoke_model_stream(
                agent, original_task, memory, **kwargs
            ):
                if isinstance(chunk, ClientResponse):
                    stream_usage += chunk.usage
                    response = chunk
                    if chunk.delta:
                        yield chunk

            if response is None:
                raise RuntimeError("No response from client")
            response.usage = stream_usage

            step_result, handoff = await self._a_finalize_step(
                agent, current_steps, original_task, memory, response
            )
            yield step_result

            if agent._hooks:
                agent._hooks.after_step(context, step_result)

            if handoff is not None:
                yield handoff
                return

            step_output = step_result.text
            step_has_tools = bool(step_result.tools_used)
            step_has_structured = bool(step_result.structured_data)

            if step_output and agent._terminate_on_text and not step_has_tools:
                final_answer = step_output
                break

            if agent._output_cls and step_has_structured and not step_has_tools:
                final_answer = ""
                break

            if agent._contains_ending_tool(step_result):
                agent._logger.debug("ending tool found, ending agent")
                break

            current_steps += 1
            original_task = ""

        if final_answer:
            agent._logger.log_panel(final_answer, title="FINAL ANSWER")

    def _initial_memory(self, agent: "Agent") -> Memory:
        return agent._memory.copy()

    def _run_single_agent(
        self,
        agent: "Agent",
        task_input: str,
        memory: Memory,
        tool_choice: Literal["auto", "required", "none", "required_first"] | list[str],
        **kwargs,
    ) -> _SingleAgentResult:
        from .agent import Plan, StepContext

        agent._logger.debug("STARTING AGENT")
        final_answer = None
        current_steps = 1
        original_task = task_input
        steps: list[StepResult] = []

        while final_answer is None and (
            agent._max_steps is None
            or (agent._max_steps and current_steps <= agent._max_steps)
        ):
            kwargs = dict(kwargs)
            kwargs["tool_choice"] = self._resolve_tool_choice(
                tool_choice, current_steps
            )

            context = StepContext(
                agent=agent,
                step_index=current_steps,
                task_input=original_task,
                memory=memory,
                tool_choice=kwargs["tool_choice"],
            )
            if agent._hooks:
                agent._hooks.before_step(context)

            agent._logger.debug(f"--- STEP {current_steps} ---")

            if agent._planning_interval and (
                current_steps == 1
                or (current_steps - 1) % agent._planning_interval == 0
            ):
                plan = self._create_plan(agent, original_task, memory)
                assert isinstance(plan, Plan)
                memory.add_turn(TextBlock(content=str(plan)), role=ROLE.ASSISTANT)
                memory.add_turn(
                    TextBlock(content="Ok, go ahead and now execute the plan."),
                    role=ROLE.USER,
                )
                agent._logger.log_panel(str(plan), title="PLAN")

            response = self._invoke_model(agent, original_task, memory, **kwargs)
            step_result, handoff = self._finalize_step(
                agent, current_steps, original_task, memory, response
            )
            steps.append(step_result)

            if agent._hooks:
                agent._hooks.after_step(context, step_result)

            step_output = step_result.text
            step_has_tools = bool(step_result.tools_used)
            step_has_structured = bool(step_result.structured_data)

            if handoff is not None:
                return _SingleAgentResult(
                    final_step=step_result,
                    handoff=handoff,
                    usage=sum_token_usage([step.usage for step in steps]),
                )

            if step_output and agent._terminate_on_text and not step_has_tools:
                final_answer = step_output
                break

            if agent._output_cls and step_has_structured and not step_has_tools:
                final_answer = ""
                break

            if agent._contains_ending_tool(step_result):
                agent._logger.debug("ending tool found, ending agent")
                break

            current_steps += 1
            original_task = ""

        if final_answer:
            agent._logger.log_panel(final_answer, title="FINAL ANSWER")

        final_step = steps[-1] if steps else None
        usage = sum_token_usage([step.usage for step in steps])
        return _SingleAgentResult(final_step=final_step, handoff=None, usage=usage)

    async def _a_run_single_agent(
        self,
        agent: "Agent",
        task_input: str,
        memory: Memory,
        tool_choice: Literal["auto", "required", "none", "required_first"] | list[str],
        **kwargs,
    ) -> _SingleAgentResult:
        from .agent import Plan, StepContext

        agent._logger.debug("STARTING AGENT")
        final_answer = None
        current_steps = 1
        original_task = task_input
        steps: list[StepResult] = []

        while final_answer is None and (
            agent._max_steps is None
            or (agent._max_steps and current_steps <= agent._max_steps)
        ):
            kwargs = dict(kwargs)
            kwargs["tool_choice"] = self._resolve_tool_choice(
                tool_choice, current_steps
            )

            context = StepContext(
                agent=agent,
                step_index=current_steps,
                task_input=original_task,
                memory=memory,
                tool_choice=kwargs["tool_choice"],
            )
            if agent._hooks:
                agent._hooks.before_step(context)

            agent._logger.debug(f"--- STEP {current_steps} ---")

            if agent._planning_interval and (
                current_steps == 1
                or (current_steps - 1) % agent._planning_interval == 0
            ):
                plan = await self._a_create_plan(agent, original_task, memory)
                assert isinstance(plan, Plan)
                memory.add_turn(TextBlock(content=str(plan)), role=ROLE.ASSISTANT)
                memory.add_turn(
                    TextBlock(content="Ok, go ahead and now execute the plan."),
                    role=ROLE.USER,
                )
                agent._logger.log_panel(str(plan), title="PLAN")

            response = await self._a_invoke_model(
                agent, original_task, memory, **kwargs
            )
            step_result, handoff = await self._a_finalize_step(
                agent, current_steps, original_task, memory, response
            )
            steps.append(step_result)

            if agent._hooks:
                agent._hooks.after_step(context, step_result)

            step_output = step_result.text
            step_has_tools = bool(step_result.tools_used)
            step_has_structured = bool(step_result.structured_data)

            if handoff is not None:
                return _SingleAgentResult(
                    final_step=step_result,
                    handoff=handoff,
                    usage=sum_token_usage([step.usage for step in steps]),
                )

            if step_output and agent._terminate_on_text and not step_has_tools:
                final_answer = step_output
                break

            if agent._output_cls and step_has_structured and not step_has_tools:
                final_answer = ""
                break

            if agent._contains_ending_tool(step_result):
                agent._logger.debug("ending tool found, ending agent")
                break

            current_steps += 1
            original_task = ""

        if final_answer:
            agent._logger.log_panel(final_answer, title="FINAL ANSWER")

        final_step = steps[-1] if steps else None
        usage = sum_token_usage([step.usage for step in steps])
        return _SingleAgentResult(final_step=final_step, handoff=None, usage=usage)

    def _resolve_tool_choice(
        self,
        tool_choice: Literal["auto", "required", "none", "required_first"] | list[str],
        current_step: int,
    ) -> Literal["auto", "required", "none"] | list[str]:
        if tool_choice == "required_first":
            return "required" if current_step == 1 else "auto"
        return tool_choice

    def _create_plan(self, agent: "Agent", task_input: str, memory: Memory) -> "Plan":
        from .agent import Plan

        prompt = agent.system_prompt + agent._planning_prompt
        client_response = agent._client.structured_response(
            input=task_input,
            tools=self._agent_tools(agent),
            tool_choice="none",
            memory=memory,
            system_prompt=prompt,
            output_cls=Plan,
        )
        return Plan(**client_response.structured_data[0].model_dump())

    async def _a_create_plan(
        self, agent: "Agent", task_input: str, memory: Memory
    ) -> "Plan":
        from .agent import Plan

        prompt = agent.system_prompt + agent._planning_prompt
        client_response = await agent._client.a_structured_response(
            input=task_input,
            tools=self._agent_tools(agent),
            tool_choice="none",
            memory=memory,
            system_prompt=prompt,
            output_cls=Plan,
        )
        return Plan(**client_response.structured_data[0].model_dump())

    def _invoke_model(
        self, agent: "Agent", task_input: str, memory: Memory, **kwargs
    ) -> ClientResponse:
        response: ClientResponse | None = None
        tools = self._agent_tools(agent)

        if agent._output_cls:
            try:
                response = agent._client.structured_response(
                    input=task_input,
                    output_cls=agent._output_cls,
                    tools=tools,
                    memory=memory,
                    system_prompt=agent.system_prompt,
                    **kwargs,
                )
            except NotImplementedError as err:
                raise ValueError(
                    f"{agent._client.__class__.__name__} does not support structured responses. "
                    "Please use a client with structured output support or remove output_cls."
                ) from err
        elif agent._stream:
            stream_usage = TokenUsage()
            for chunk in agent._client.stream_invoke(
                input=task_input,
                tools=tools,
                memory=memory,
                system_prompt=agent.system_prompt,
                **kwargs,
            ):
                stream_usage += chunk.usage
                response = chunk
            if response is not None:
                response.usage = stream_usage
        else:
            response = agent._client.invoke(
                input=task_input,
                tools=tools,
                memory=memory,
                system_prompt=agent.system_prompt,
                **kwargs,
            )

        if response is None:
            raise RuntimeError("No response from client")
        return response

    def _invoke_model_stream(
        self, agent: "Agent", task_input: str, memory: Memory, **kwargs
    ) -> "Generator[ClientResponse, None, None]":
        tools = self._agent_tools(agent)

        if agent._output_cls:
            try:
                yield agent._client.structured_response(
                    input=task_input,
                    output_cls=agent._output_cls,
                    tools=tools,
                    memory=memory,
                    system_prompt=agent.system_prompt,
                    **kwargs,
                )
                return
            except NotImplementedError as err:
                raise ValueError(
                    f"{agent._client.__class__.__name__} does not support structured responses. "
                    "Please use a client with structured output support or remove output_cls."
                ) from err

        if agent._stream:
            yield from agent._client.stream_invoke(
                input=task_input,
                tools=tools,
                memory=memory,
                system_prompt=agent.system_prompt,
                **kwargs,
            )
            return

        yield agent._client.invoke(
            input=task_input,
            tools=tools,
            memory=memory,
            system_prompt=agent.system_prompt,
            **kwargs,
        )

    async def _a_invoke_model(
        self, agent: "Agent", task_input: str, memory: Memory, **kwargs
    ) -> ClientResponse:
        response: ClientResponse | None = None
        tools = self._agent_tools(agent)

        if agent._output_cls:
            try:
                response = await agent._client.a_structured_response(
                    input=task_input,
                    output_cls=agent._output_cls,
                    tools=tools,
                    memory=memory,
                    system_prompt=agent.system_prompt,
                    **kwargs,
                )
            except NotImplementedError as err:
                raise ValueError(
                    f"{agent._client.__class__.__name__} does not support async structured responses. "
                    "Please use a client with structured output support or remove output_cls."
                ) from err
        elif agent._stream:
            stream_usage = TokenUsage()
            async for chunk in agent._client.a_stream_invoke(
                input=task_input,
                tools=tools,
                memory=memory,
                system_prompt=agent.system_prompt,
                **kwargs,
            ):
                stream_usage += chunk.usage
                response = chunk
            if response is not None:
                response.usage = stream_usage
        else:
            response = await agent._client.a_invoke(
                input=task_input,
                tools=tools,
                memory=memory,
                system_prompt=agent.system_prompt,
                **kwargs,
            )

        if response is None:
            raise RuntimeError("No response from client")
        return response

    async def _a_invoke_model_stream(
        self, agent: "Agent", task_input: str, memory: Memory, **kwargs
    ) -> "AsyncGenerator[ClientResponse, None]":
        tools = self._agent_tools(agent)

        if agent._output_cls:
            try:
                yield await agent._client.a_structured_response(
                    input=task_input,
                    output_cls=agent._output_cls,
                    tools=tools,
                    memory=memory,
                    system_prompt=agent.system_prompt,
                    **kwargs,
                )
                return
            except NotImplementedError as err:
                raise ValueError(
                    f"{agent._client.__class__.__name__} does not support async structured responses. "
                    "Please use a client with structured output support or remove output_cls."
                ) from err

        if agent._stream:
            async for chunk in agent._client.a_stream_invoke(
                input=task_input,
                tools=tools,
                memory=memory,
                system_prompt=agent.system_prompt,
                **kwargs,
            ):
                yield chunk
            return

        yield await agent._client.a_invoke(
            input=task_input,
            tools=tools,
            memory=memory,
            system_prompt=agent.system_prompt,
            **kwargs,
        )

    def _finalize_step(
        self,
        agent: "Agent",
        current_step: int,
        task_input: str,
        memory: Memory,
        response: ClientResponse,
    ) -> tuple["StepResult", HandoffRequest | None]:
        from .agent import StepResult

        handoff_calls = self._get_handoff_calls(agent, response.function_calls)

        if task_input:
            memory.add_turn(TextBlock(content=task_input), role=ROLE.USER)

        if response.text:
            memory.add_turn(TextBlock(content=response.text), role=ROLE.ASSISTANT)
        elif response.structured_data:
            memory.add_turn(response.content, role=ROLE.ASSISTANT)

        if response.function_calls:
            memory.add_turn(response.function_calls, role=ROLE.ASSISTANT)

        handoff = self._extract_handoff(agent, handoff_calls)
        tool_results: list[FunctionCallResultBlock] = []
        if handoff is None:
            for tool_call in response.function_calls:
                tool_results.append(self._execute_tool(tool_call, agent))
        else:
            tool_results.append(self._handoff_result_block(handoff_calls[0], handoff))

        for tool_result in tool_results:
            memory.add_turn(tool_result, role=ROLE.TOOL)

        step_action = StepResult(
            index=current_step,
            content=response.content + tool_results,
            usage=response.usage,
        )
        return step_action, handoff

    async def _a_finalize_step(
        self,
        agent: "Agent",
        current_step: int,
        task_input: str,
        memory: Memory,
        response: ClientResponse,
    ) -> tuple["StepResult", HandoffRequest | None]:
        from .agent import StepResult

        handoff_calls = self._get_handoff_calls(agent, response.function_calls)

        if task_input:
            memory.add_turn(TextBlock(content=task_input), role=ROLE.USER)

        if response.text:
            memory.add_turn(TextBlock(content=response.text), role=ROLE.ASSISTANT)
        elif response.structured_data:
            memory.add_turn(response.content, role=ROLE.ASSISTANT)

        if response.function_calls:
            memory.add_turn(response.function_calls, role=ROLE.ASSISTANT)

        handoff = self._extract_handoff(agent, handoff_calls)
        tool_results: list[FunctionCallResultBlock] = []
        if handoff is None:
            for tool_call in response.function_calls:
                tool_results.append(await self._a_execute_tool(tool_call, agent))
        else:
            tool_results.append(self._handoff_result_block(handoff_calls[0], handoff))

        for tool_result in tool_results:
            memory.add_turn(tool_result, role=ROLE.TOOL)

        step_action = StepResult(
            index=current_step,
            content=response.content + tool_results,
            usage=response.usage,
        )
        return step_action, handoff

    def _agent_tools(self, agent: "Agent") -> list[Tool]:
        return [*agent._tools, *self._handoff_tools(agent)]

    def _handoff_tools(self, agent: "Agent") -> list[Tool]:
        tools = []
        for target in agent._handoffs:
            tools.append(
                Tool(
                    name=self._handoff_tool_name(target),
                    description=f"Transfer control to {target.name}.",
                    properties={
                        "task_input": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    required=["task_input"],
                )
            )
        return tools

    def _extract_handoff(
        self,
        agent: "Agent",
        function_calls: list[FunctionCallBlock],
    ) -> HandoffRequest | None:
        handoff_calls = self._get_handoff_calls(agent, function_calls)
        if not handoff_calls:
            return None
        if len(handoff_calls) > 1:
            raise ValueError("Only one handoff can be requested per step")

        handoff_call = handoff_calls[0]
        target_name = handoff_call.name.removeprefix("transfer_to_")
        task_input = handoff_call.arguments.get("task_input")
        if not isinstance(task_input, str) or not task_input:
            raise ValueError("Handoff requires a non-empty 'task_input'")

        reason = handoff_call.arguments.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("Handoff 'reason' must be a string")

        return HandoffRequest(target=target_name, task_input=task_input, reason=reason)

    def _get_handoff_calls(
        self,
        agent: "Agent",
        function_calls: list[FunctionCallBlock],
    ) -> list[FunctionCallBlock]:
        return [
            call
            for call in function_calls
            if any(
                call.name == self._handoff_tool_name(target)
                for target in agent._handoffs
            )
        ]

    def _handoff_result_block(
        self,
        function_call: FunctionCallBlock,
        handoff: HandoffRequest,
    ) -> FunctionCallResultBlock:
        return FunctionCallResultBlock(
            id=function_call.id,
            tool=function_call.tool,
            result=f"Transfer to {handoff.target} completed",
        )

    def _resolve_handoff(self, agent: "Agent", request: HandoffRequest) -> "Agent":
        for target in agent._handoffs:
            if target.name == request.target:
                return target
        raise ValueError(
            f"Unknown handoff target '{request.target}' for agent '{agent.name}'"
        )

    def _handoff_tool_name(self, agent: "Agent") -> str:
        return f"transfer_to_{agent.name}"

    def _execute_tool(
        self, function_call: FunctionCallBlock, agent: "Agent"
    ) -> FunctionCallResultBlock:
        with tool_span(f"Tool {function_call.tool.name}") as current_tool_span:
            current_tool_span.set_attribute(
                "tool_arguments", str(function_call.arguments)
            )
            result = function_call.tool(**function_call.arguments)

            if inspect.iscoroutine(result):
                result = AsyncExecutor.get_instance().run(result)

            if result:
                current_tool_span.set_attribute("tool_result", result)
                agent._logger.log_panel(
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
        self, function_call: FunctionCallBlock, agent: "Agent"
    ) -> FunctionCallResultBlock:
        with tool_span(f"Tool {function_call.tool.name}") as current_tool_span:
            current_tool_span.set_attribute(
                "tool_arguments", str(function_call.arguments)
            )
            result = function_call.tool(**function_call.arguments)

            if inspect.iscoroutine(result):
                result = await result

            if result:
                current_tool_span.set_attribute("tool_result", result)
                agent._logger.log_panel(
                    result,
                    title=f"TOOL {function_call.tool.name.upper()} RESULT",
                    subtitle="args: " + str(function_call.arguments),
                )
            return FunctionCallResultBlock(
                id=function_call.id,
                tool=function_call.tool,
                result=result,
            )
