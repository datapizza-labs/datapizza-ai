import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

from datapizza.core.cache import Cache
from datapizza.core.clients import Client, ClientResponse
from datapizza.core.clients.models import TokenUsage
from datapizza.memory import Memory
from datapizza.tools import Tool
from datapizza.type import (
    Block,
    FunctionCallBlock,
    Model,
    StructuredBlock,
    TextBlock,
    ThoughtBlock,
)

from anthropic import Anthropic, AsyncAnthropic

from .memory_adapter import AnthropicMemoryAdapter


class AnthropicClient(Client):
    """A client for interacting with the Anthropic API (Claude).

    This class provides methods for invoking the Anthropic API to generate responses
    based on given input data. It extends the Client class.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-latest",
        system_prompt: str = "",
        temperature: float | None = None,
        cache: Cache | None = None,
    ):
        """
        Args:
            api_key: The API key for the Anthropic API.
            model: The model to use for the Anthropic API.
            system_prompt: The system prompt to use for the Anthropic API.
            temperature: The temperature to use for the Anthropic API.
            cache: The cache to use for the Anthropic API.
        """
        if temperature and not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")

        super().__init__(
            model_name=model,
            system_prompt=system_prompt,
            temperature=temperature,
            cache=cache,
        )
        self.api_key = api_key
        self.memory_adapter = AnthropicMemoryAdapter()
        self._set_client()

    def _set_client(self):
        if not self.client:
            self.client = Anthropic(api_key=self.api_key)

    def _set_a_client(self):
        if not self.a_client:
            self.a_client = AsyncAnthropic(api_key=self.api_key)

    def _convert_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Convert tools to Anthropic tool format"""
        anthropic_tools = []
        for tool in tools:
            anthropic_tool = {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": {
                    "type": "object",
                    "properties": tool.properties,
                    "required": tool.required,
                },
            }
            anthropic_tools.append(anthropic_tool)
        return anthropic_tools

    def _convert_tool_choice(
        self, tool_choice: Literal["auto", "required", "none"] | list[str]
    ) -> dict | Literal["auto", "required", "none"]:
        if isinstance(tool_choice, list) and len(tool_choice) > 1:
            raise NotImplementedError(
                "multiple function names is not supported by Anthropic"
            )
        elif isinstance(tool_choice, list):
            return {
                "type": "tool",
                "name": tool_choice[0],
            }
        elif tool_choice == "required":
            return {"type": "any"}
        elif tool_choice == "auto":
            return {"type": "auto"}
        else:
            return tool_choice

    def _response_to_client_response(
        self, response, tool_map: dict[str, Tool] | None = None
    ) -> ClientResponse:
        """Convert Anthropic response to ClientResponse"""
        blocks = []

        if hasattr(response, "content") and response.content:
            if isinstance(
                response.content, list
            ):  # Claude 3 returns a list of content blocks
                for content_block in response.content:
                    if content_block.type == "text":
                        blocks.append(TextBlock(content=content_block.text))
                    elif content_block.type == "thinking":
                        # Summarized thinking content
                        blocks.append(ThoughtBlock(content=content_block.thinking))
                    elif content_block.type == "tool_use":
                        tool = tool_map.get(content_block.name) if tool_map else None
                        if not tool:
                            raise ValueError(f"Tool {content_block.name} not found")

                        blocks.append(
                            FunctionCallBlock(
                                id=content_block.id,
                                name=content_block.name,
                                arguments=content_block.input,
                                tool=tool,
                            )
                        )
            else:  # Handle as string for compatibility
                blocks.append(TextBlock(content=str(response.content)))

        stop_reason = response.stop_reason if hasattr(response, "stop_reason") else None

        return ClientResponse(
            content=blocks,
            stop_reason=stop_reason,
            usage=TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                cached_tokens=response.usage.cache_read_input_tokens,
            ),
        )

    def _usage_from_anthropic_response(self, response: Any) -> TokenUsage:
        usage = response.usage
        return TokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            cached_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        )

    def _new_stream_state(self) -> dict[str, Any]:
        return {
            "message_text": "",
            "thought_text": "",
            "tool_calls": {},
            "input_tokens": 0,
            "output_tokens": 0,
            "stop_reason": None,
        }

    def _consume_stream_event(self, state: dict[str, Any], chunk: Any) -> str:
        text_delta = ""

        if chunk.type == "message_start":
            if getattr(chunk, "message", None) and getattr(
                chunk.message, "usage", None
            ):
                state["input_tokens"] = chunk.message.usage.input_tokens or 0
            return text_delta

        if chunk.type == "content_block_start":
            content_block = getattr(chunk, "content_block", None)
            if content_block and getattr(content_block, "type", None) == "tool_use":
                state["tool_calls"][chunk.index] = {
                    "id": getattr(content_block, "id", None),
                    "name": getattr(content_block, "name", None),
                    "input_json_chunks": [],
                    "parsed_input": (
                        dict(content_block.input)
                        if getattr(content_block, "input", None)
                        else None
                    ),
                }
            return text_delta

        if chunk.type == "content_block_delta":
            delta = getattr(chunk, "delta", None)
            if not delta:
                return text_delta

            if getattr(delta, "text", None):
                text_delta = delta.text
                state["message_text"] += delta.text
                return text_delta

            if getattr(delta, "thinking", None):
                state["thought_text"] += delta.thinking
                return text_delta

            if getattr(delta, "partial_json", None):
                tool_state = state["tool_calls"].setdefault(
                    chunk.index,
                    {
                        "id": None,
                        "name": None,
                        "input_json_chunks": [],
                        "parsed_input": None,
                    },
                )
                tool_state["input_json_chunks"].append(delta.partial_json)
                return text_delta

            return text_delta

        if chunk.type == "message_delta":
            if getattr(chunk, "usage", None):
                state["output_tokens"] = max(
                    state["output_tokens"],
                    chunk.usage.output_tokens or 0,
                )

            delta = getattr(chunk, "delta", None)
            if delta and hasattr(delta, "stop_reason"):
                state["stop_reason"] = delta.stop_reason

        return text_delta

    def _build_streamed_tool_call_blocks(
        self,
        tool_calls: dict[int, dict[str, Any]],
        tool_map: dict[str, Tool],
    ) -> list[FunctionCallBlock]:
        blocks: list[FunctionCallBlock] = []

        for _, tool_state in sorted(tool_calls.items()):
            tool_name = tool_state.get("name")
            if not tool_name:
                continue

            tool = tool_map.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")

            parsed_input = tool_state.get("parsed_input")
            if parsed_input is None:
                input_json = "".join(tool_state.get("input_json_chunks", []))
                parsed_input = json.loads(input_json) if input_json else {}

            blocks.append(
                FunctionCallBlock(
                    id=tool_state.get("id") or tool_name,
                    name=tool_name,
                    arguments=parsed_input,
                    tool=tool,
                )
            )

        return blocks

    def _build_stream_final_content(
        self,
        state: dict[str, Any],
        tool_map: dict[str, Tool],
    ) -> list[ThoughtBlock | TextBlock | FunctionCallBlock]:
        content: list[ThoughtBlock | TextBlock | FunctionCallBlock] = []

        if state["thought_text"]:
            content.append(ThoughtBlock(content=state["thought_text"]))
        if state["message_text"]:
            content.append(TextBlock(content=state["message_text"]))
        content.extend(
            self._build_streamed_tool_call_blocks(state["tool_calls"], tool_map)
        )
        return content

    def _structured_messages(
        self, input: list[Block], memory: Memory | None
    ) -> list[dict]:
        messages = self._memory_to_contents(None, input, memory)
        return [m for m in messages if m.get("role") != "model"]

    def _structured_request_params(
        self,
        *,
        messages: list[dict],
        max_tokens: int | None,
        temperature: float | None,
        system_prompt: str | None,
        tools: list[Tool],
        tool_choice: Literal["auto", "required", "none"] | list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or 2048,
            **kwargs,
        }
        if temperature:
            params["temperature"] = temperature
        if system_prompt:
            params["system"] = system_prompt
        if tools:
            params["tools"] = self._convert_tools(tools)
            params["tool_choice"] = self._convert_tool_choice(tool_choice)
        return params

    def _invoke(
        self,
        *,
        input: str,
        tools: list[Tool] | None,
        memory: Memory | None,
        tool_choice: Literal["auto", "required", "none"] | list[str],
        temperature: float | None,
        max_tokens: int | None,
        system_prompt: str | None,
        **kwargs,
    ) -> ClientResponse:
        """Implementation of the abstract _invoke method for Anthropic"""
        if tools is None:
            tools = []
        client = self._get_client()
        messages = self._memory_to_contents(None, input, memory)
        # remove the model from the messages
        messages = [message for message in messages if message.get("role") != "model"]

        tool_map = {tool.name: tool for tool in tools}

        request_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or 2048,
            **kwargs,
        }

        if temperature:
            request_params["temperature"] = temperature

        if system_prompt:
            request_params["system"] = system_prompt

        if tools:
            request_params["tools"] = self._convert_tools(tools)
            request_params["tool_choice"] = self._convert_tool_choice(tool_choice)

        response = client.messages.create(**request_params)
        return self._response_to_client_response(response, tool_map)

    async def _a_invoke(
        self,
        *,
        input: str,
        tools: list[Tool] | None,
        memory: Memory | None,
        tool_choice: Literal["auto", "required", "none"] | list[str],
        temperature: float | None,
        max_tokens: int | None,
        system_prompt: str | None,
        **kwargs,
    ) -> ClientResponse:
        if tools is None:
            tools = []
        client = self._get_a_client()
        messages = self._memory_to_contents(None, input, memory)
        # remove the model from the messages
        messages = [message for message in messages if message.get("role") != "model"]

        tool_map = {tool.name: tool for tool in tools}

        request_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or 2048,
            **kwargs,
        }

        if temperature:
            request_params["temperature"] = temperature

        if system_prompt:
            request_params["system"] = system_prompt

        if tools:
            request_params["tools"] = self._convert_tools(tools)
            request_params["tool_choice"] = self._convert_tool_choice(tool_choice)

        response = await client.messages.create(**request_params)
        return self._response_to_client_response(response, tool_map)

    def _stream_invoke(
        self,
        input: str,
        tools: list[Tool] | None,
        memory: Memory | None,
        tool_choice: Literal["auto", "required", "none"] | list[str],
        temperature: float | None,
        max_tokens: int | None,
        system_prompt: str | None,
        **kwargs,
    ) -> Iterator[ClientResponse]:
        """Implementation of the abstract _stream_invoke method for Anthropic"""
        if tools is None:
            tools = []
        messages = self._memory_to_contents(None, input, memory)
        client = self._get_client()
        tool_map = {tool.name: tool for tool in tools}

        request_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens or 2048,
            **kwargs,
        }

        if temperature:
            request_params["temperature"] = temperature

        if system_prompt:
            request_params["system"] = system_prompt

        if tools:
            request_params["tools"] = self._convert_tools(tools)
            request_params["tool_choice"] = self._convert_tool_choice(tool_choice)

        stream = client.messages.create(**request_params)
        state = self._new_stream_state()

        for chunk in stream:
            text_delta = self._consume_stream_event(state, chunk)
            if text_delta:
                yield ClientResponse(
                    content=[
                        ThoughtBlock(content=state["thought_text"]),
                        TextBlock(content=state["message_text"]),
                    ],
                    delta=text_delta,
                    stop_reason=state["stop_reason"],
                )

        yield ClientResponse(
            content=self._build_stream_final_content(state, tool_map),
            delta="",
            stop_reason=state["stop_reason"],
            usage=TokenUsage(
                prompt_tokens=state["input_tokens"],
                completion_tokens=state["output_tokens"],
                cached_tokens=0,
            ),
        )

    async def _a_stream_invoke(
        self,
        input: str,
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[ClientResponse]:
        """Implementation of the abstract _a_stream_invoke method for Anthropic"""
        if tools is None:
            tools = []
        messages = self._memory_to_contents(None, input, memory)
        client = self._get_a_client()
        tool_map = {tool.name: tool for tool in tools}

        request_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens or 2048,
            **kwargs,
        }
        if temperature:
            request_params["temperature"] = temperature
        if system_prompt:
            request_params["system"] = system_prompt

        if max_tokens:
            request_params["max_tokens"] = max_tokens

        if tools:
            request_params["tools"] = self._convert_tools(tools)
            request_params["tool_choice"] = self._convert_tool_choice(tool_choice)

        stream = await client.messages.create(**request_params)
        state = self._new_stream_state()

        async for chunk in stream:
            text_delta = self._consume_stream_event(state, chunk)
            if text_delta:
                yield ClientResponse(
                    content=[
                        ThoughtBlock(content=state["thought_text"]),
                        TextBlock(content=state["message_text"]),
                    ],
                    delta=text_delta,
                    stop_reason=state["stop_reason"],
                )

        yield ClientResponse(
            content=self._build_stream_final_content(state, tool_map),
            delta="",
            stop_reason=state["stop_reason"],
            usage=TokenUsage(
                prompt_tokens=state["input_tokens"],
                completion_tokens=state["output_tokens"],
                cached_tokens=0,
            ),
        )

    def _structured_response(
        self,
        input: list[Block],
        output_cls: type[Model],
        memory: Memory | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        **kwargs: Any,
    ) -> ClientResponse:
        if tools is None:
            tools = []
        client = self._get_client()
        messages = self._structured_messages(input, memory)
        params = self._structured_request_params(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        if output_cls == {"type": "json_object"}:
            params["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object", "additionalProperties": True},
                }
            }
            response = client.messages.create(**params)
            text: str | None = None
            for block in response.content:
                if block.type == "text":
                    text = block.text
                    break
            if text is None:
                raise ValueError("No text block in Anthropic structured response")
            data = json.loads(text)
            return ClientResponse(
                content=[StructuredBlock(content=data)],
                stop_reason=response.stop_reason,
                usage=self._usage_from_anthropic_response(response),
            )

        response = client.messages.parse(**params, output_format=output_cls)
        parsed = response.parsed_output
        if parsed is None:
            raise ValueError("No parsed_output in Anthropic structured response")
        return ClientResponse(
            content=[StructuredBlock(content=parsed)],
            stop_reason=response.stop_reason,
            usage=self._usage_from_anthropic_response(response),
        )

    async def _a_structured_response(
        self,
        input: list[Block],
        output_cls: type[Model],
        memory: Memory | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        **kwargs: Any,
    ) -> ClientResponse:
        if tools is None:
            tools = []
        client = self._get_a_client()
        messages = self._structured_messages(input, memory)
        params = self._structured_request_params(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        if output_cls == {"type": "json_object"}:
            params["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object", "additionalProperties": True},
                }
            }
            response = await client.messages.create(**params)
            text: str | None = None
            for block in response.content:
                if block.type == "text":
                    text = block.text
                    break
            if text is None:
                raise ValueError("No text block in Anthropic structured response")
            data = json.loads(text)
            return ClientResponse(
                content=[StructuredBlock(content=data)],
                stop_reason=response.stop_reason,
                usage=self._usage_from_anthropic_response(response),
            )

        response = await client.messages.parse(**params, output_format=output_cls)
        parsed = response.parsed_output
        if parsed is None:
            raise ValueError("No parsed_output in Anthropic structured response")
        return ClientResponse(
            content=[StructuredBlock(content=parsed)],
            stop_reason=response.stop_reason,
            usage=self._usage_from_anthropic_response(response),
        )
