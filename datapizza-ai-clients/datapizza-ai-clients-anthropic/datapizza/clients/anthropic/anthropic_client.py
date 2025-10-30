from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

from datapizza.core.cache import Cache
from datapizza.core.clients import Client, ClientResponse
from datapizza.core.clients.models import TokenUsage
from datapizza.memory import Memory
from datapizza.tools import Tool
from datapizza.type import FunctionCallBlock, TextBlock, ThoughtBlock

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

        input_tokens = 0
        output_tokens = 0
        message_text = ""
        thought_text = ""

        for chunk in stream:
            if (
                chunk.type == "content_block_delta"
                and hasattr(chunk, "delta")
                and chunk.delta
            ):
                if hasattr(chunk.delta, "text") and chunk.delta.text:
                    message_text += chunk.delta.text
                    yield ClientResponse(
                        content=[
                            ThoughtBlock(content=thought_text),
                            TextBlock(content=message_text),
                        ],
                        delta=chunk.delta.text,
                    )
                elif hasattr(chunk.delta, "thinking") and chunk.delta.thinking:
                    thought_text += chunk.delta.thinking

            if chunk.type == "message_start":
                input_tokens = (
                    chunk.message.usage.input_tokens if chunk.message.usage else 0
                )

            if chunk.type == "message_delta":
                output_tokens = max(
                    output_tokens, chunk.usage.output_tokens if chunk.usage else 0
                )

        yield ClientResponse(
            content=[
                ThoughtBlock(content=thought_text),
                TextBlock(content=message_text),
            ],
            delta="",
            stop_reason="end_turn",
            usage=TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
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

        input_tokens = 0
        output_tokens = 0
        message_text = ""
        thought_text = ""

        async for chunk in stream:
            if (
                chunk.type == "content_block_delta"
                and hasattr(chunk, "delta")
                and chunk.delta
            ):
                if hasattr(chunk.delta, "text") and chunk.delta.text:
                    message_text += chunk.delta.text
                    yield ClientResponse(
                        content=[
                            ThoughtBlock(content=thought_text),
                            TextBlock(content=message_text),
                        ],
                        delta=chunk.delta.text,
                    )
                elif hasattr(chunk.delta, "thinking") and chunk.delta.thinking:
                    thought_text += chunk.delta.thinking

            if chunk.type == "message_start":
                input_tokens = (
                    chunk.message.usage.input_tokens if chunk.message.usage else 0
                )

            if chunk.type == "message_delta":
                output_tokens = max(
                    output_tokens, chunk.usage.output_tokens if chunk.usage else 0
                )

        yield ClientResponse(
            content=[
                ThoughtBlock(content=thought_text),
                TextBlock(content=message_text),
            ],
            delta="",
            stop_reason="end_turn",
            usage=TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                cached_tokens=0,
            ),
        )

    def _structured_response(
        self,
        *args,
        **kwargs,
    ) -> ClientResponse:
        raise NotImplementedError("Anthropic does not support structured responses")

    async def _a_structured_response(self, *args, **kwargs):
        raise NotImplementedError("Anthropic does not support structured responses")
