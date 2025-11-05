import json
from collections.abc import AsyncIterator, Iterator
from typing import Literal

import httpx
from datapizza.core.cache import Cache
from datapizza.core.clients import Client, ClientResponse
from datapizza.core.clients.models import TokenUsage
from datapizza.memory import Memory
from datapizza.tools.tools import Tool
from datapizza.type import (
    FunctionCallBlock,
    Media,
    MediaBlock,
    Model,
    StructuredBlock,
    TextBlock,
)
from openai import AsyncOpenAI, OpenAI

from .memory_adapter import OpenAILikeMemoryAdapter


class OpenAILikeClient(Client):
    """A client for interacting with the OpenAI API.

    This class provides methods for invoking the OpenAI API to generate responses
    based on given input data. It extends the Client class.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        system_prompt: str = "",
        temperature: float | None = None,
        cache: Cache | None = None,
        base_url: str | httpx.URL | None = None,
    ):
        if temperature and not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")

        super().__init__(
            model_name=model,
            system_prompt=system_prompt,
            temperature=temperature,
            cache=cache,
        )

        self.base_url = base_url
        self.api_key = api_key
        self.memory_adapter = OpenAILikeMemoryAdapter()
        self._set_client()

    def _set_client(self):
        if not self.client:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _set_a_client(self):
        if not self.a_client:
            self.a_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _response_to_client_response(
        self, response, tool_map: dict[str, Tool] | None
    ) -> ClientResponse:
        blocks = []
        for choice in response.choices:
            if choice.message.content:
                blocks.append(TextBlock(content=choice.message.content))

            if choice.message.tool_calls and tool_map:
                for tool_call in choice.message.tool_calls:
                    tool = tool_map.get(tool_call.function.name)

                    if not tool:
                        raise ValueError(f"Tool {tool_call.function.name} not found")

                    blocks.append(
                        FunctionCallBlock(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=json.loads(tool_call.function.arguments),
                            tool=tool,
                        )
                    )

            # Handle media content if present
            if hasattr(choice.message, "media") and choice.message.media:
                for media_item in choice.message.media:
                    media = Media(
                        media_type=media_item.type,
                        source_type="url" if media_item.source_url else "base64",
                        source=media_item.source_url or media_item.data,
                        detail=getattr(media_item, "detail", "high"),
                    )
                    blocks.append(MediaBlock(media=media))

        return ClientResponse(
            content=blocks,
            stop_reason=response.choices[0].finish_reason,
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens
                if response.usage.prompt_tokens_details
                else 0 or 0,
            ),
        )

    def _convert_tools(self, tools: Tool) -> dict:
        """Convert tools to OpenAI function format"""
        return {"type": "function", "function": tools.schema}

    def _convert_tool_choice(
        self, tool_choice: Literal["auto", "required", "none"] | list[str]
    ) -> dict | Literal["auto", "required", "none"]:
        if isinstance(tool_choice, list) and len(tool_choice) > 1:
            raise NotImplementedError(
                "multiple function names is not supported by OpenAI"
            )
        elif isinstance(tool_choice, list):
            return {
                "type": "function",
                "function": {"name": tool_choice[0]},
            }
        else:
            return tool_choice

    def _invoke(
        self,
        *,
        input: str,
        tools: list[Tool] | None,
        memory: Memory | None,
        tool_choice: Literal["auto", "required", "none"] | list[str],
        temperature: float | None,
        max_tokens: int,
        system_prompt: str | None,
        **kwargs,
    ) -> ClientResponse:
        if tools is None:
            tools = []
        messages = self._memory_to_contents(system_prompt, input, memory)

        tool_map = {tool.name: tool for tool in tools}

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "max_completion_tokens": max_tokens,
            **kwargs,
        }
        if temperature:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        client: OpenAI = self._get_client()
        response = client.chat.completions.create(**kwargs)
        return self._response_to_client_response(response, tool_map)

    async def _a_invoke(
        self,
        *,
        input: str,
        tools: list[Tool] | None,
        memory: Memory | None,
        tool_choice: Literal["auto", "required", "none"] | list[str],
        temperature: float | None,
        max_tokens: int,
        system_prompt: str | None,
        **kwargs,
    ) -> ClientResponse:
        if tools is None:
            tools = []
        messages = self._memory_to_contents(system_prompt, input, memory)

        tool_map = {tool.name: tool for tool in tools}

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "max_completion_tokens": max_tokens,
            **kwargs,
        }
        if temperature:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        a_client = self._get_a_client()
        response = await a_client.chat.completions.create(**kwargs)
        return self._response_to_client_response(response, tool_map)

    def _stream_invoke(
        self,
        input: str,
        tools: list[Tool] | None,
        memory: Memory | None,
        tool_choice: Literal["auto", "required", "none"] | list[str],
        temperature: float | None,
        max_tokens: int,
        system_prompt: str | None,
        **kwargs,
    ) -> Iterator[ClientResponse]:
        if tools is None:
            tools = []
        messages = self._memory_to_contents(system_prompt, input, memory)
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "max_completion_tokens": max_tokens,
            "stream_options": {"include_usage": True},
            **kwargs,
        }
        if temperature:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        response = self.client.chat.completions.create(**kwargs)
        message_content = ""
        usage = TokenUsage()
        finish_reason = None

        for chunk in response:
            usage += TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens if chunk.usage else 0 or 0,
                completion_tokens=chunk.usage.completion_tokens
                if chunk.usage
                else 0 or 0,
                cached_tokens=getattr(
                    getattr(chunk.usage, "prompt_tokens_details", None),
                    "cached_tokens",
                    0,
                )
                or 0,
            )

            if len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

            delta_content = delta.content if delta and delta.content else ""
            message_content = message_content + delta_content
            yield ClientResponse(
                content=[TextBlock(content=message_content)],
                delta=delta_content,
                stop_reason=finish_reason or None,
            )
        yield ClientResponse(
            content=[TextBlock(content=message_content)],
            stop_reason=finish_reason or None,
            usage=usage,
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
        if tools is None:
            tools = []
        messages = self._memory_to_contents(system_prompt, input, memory)
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "max_completion_tokens": max_tokens,
            "stream_options": {"include_usage": True},
            **kwargs,
        }
        if temperature:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        a_client = self._get_a_client()
        message_content = ""
        usage = TokenUsage()
        finish_reason = None

        async for chunk in await a_client.chat.completions.create(**kwargs):
            usage += TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens if chunk.usage else 0 or 0,
                completion_tokens=chunk.usage.completion_tokens
                if chunk.usage
                else 0 or 0,
                cached_tokens=getattr(
                    getattr(chunk.usage, "prompt_tokens_details", {}),
                    "cached_tokens",
                    0,
                )
                or 0,
            )

            if len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

            delta_content = delta.content if delta and delta.content else ""
            message_content = message_content + delta_content

            yield ClientResponse(
                content=[TextBlock(content=message_content)],
                delta=delta_content,
                stop_reason=finish_reason or None,
            )
        yield ClientResponse(
            content=[TextBlock(content=message_content)],
            stop_reason=finish_reason or None,
            usage=usage,
        )

    def _structured_response(
        self,
        input: str,
        output_cls: type[Model],
        memory: Memory | None,
        temperature: float | None,
        max_tokens: int,
        system_prompt: str | None,
        tools: list[Tool] | None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        **kwargs,
    ) -> ClientResponse:
        # Add system message to enforce JSON output
        messages = self._memory_to_contents(system_prompt, input, memory)

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "response_format": output_cls,
            "max_completion_tokens": max_tokens,
            **kwargs,
        }
        if temperature:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)
            # Structured response needs strict mode and no additional properties
            for tool in kwargs["tools"]:
                tool["function"]["strict"] = True
                tool["function"]["parameters"]["additionalProperties"] = False

        response = self.client.beta.chat.completions.parse(**kwargs)

        stop_reason = response.choices[0].finish_reason

        if not response.choices[0].message.content:
            raise ValueError("No content in response")

        if hasattr(output_cls, "model_validate_json"):
            structured_data = output_cls.model_validate_json(
                response.choices[0].message.content
            )
        else:
            structured_data = json.loads(response.choices[0].message.content)
        return ClientResponse(
            content=[StructuredBlock(content=structured_data)],
            stop_reason=stop_reason,
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens
                if response.usage.prompt_tokens_details
                else 0 or 0,
            ),
        )

    async def _a_structured_response(
        self,
        input: str,
        output_cls: type[Model],
        memory: Memory | None,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        **kwargs,
    ):
        messages = self._memory_to_contents(system_prompt, input, memory)

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "response_format": output_cls,
            "max_completion_tokens": max_tokens,
            **kwargs,
        }
        if temperature:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)
            # Structured response needs strict mode and no additional properties
            for tool in kwargs["tools"]:
                tool["function"]["strict"] = True
                tool["function"]["parameters"]["additionalProperties"] = False

        a_client = self._get_a_client()
        response = await a_client.beta.chat.completions.parse(**kwargs)

        stop_reason = response.choices[0].finish_reason
        if hasattr(output_cls, "model_validate_json"):
            structured_data = output_cls.model_validate_json(
                response.choices[0].message.content
            )
        else:
            structured_data = json.loads(response.choices[0].message.content)
        return ClientResponse(
            content=[StructuredBlock(content=structured_data)],
            stop_reason=stop_reason,
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens
                if response.usage.prompt_tokens_details
                else 0 or 0,
            ),
        )
