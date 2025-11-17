import json
from collections.abc import AsyncIterator, Iterator
from typing import Literal

import httpx
from datapizza.core.cache import Cache
from datapizza.core.clients import Client, ClientResponse
from datapizza.core.clients.models import TokenUsage
from datapizza.memory import Memory
from datapizza.tools import Tool
from datapizza.type import (
    FunctionCallBlock,
    Model,
    StructuredBlock,
    TextBlock,
    ThoughtBlock,
)

from openai import (
    AsyncOpenAI,
    AzureOpenAI,
    OpenAI,
)
from openai.types.responses import (
    ParsedResponseOutputMessage,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseTextDeltaEvent,
)

from .memory_adapter import OpenAIMemoryAdapter


class OpenAIClient(Client):
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
        organization: str | None = None,
        project: str | None = None,
        webhook_secret: str | None = None,
        websocket_base_url: str | httpx.URL | None = None,
        timeout: float | httpx.Timeout | None = None,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, object] | None = None,
        http_client: httpx.Client | None = None,
    ):
        """
        Args:
            api_key: The API key for the OpenAI API.
            model: The model to use for the OpenAI API.
            system_prompt: The system prompt to use for the OpenAI API.
            temperature: The temperature to use for the OpenAI API.
            cache: The cache to use for the OpenAI API.
            base_url: The base URL for the OpenAI API.
            organization: The organization ID for the OpenAI API.
            project: The project ID for the OpenAI API.
            webhook_secret: The webhook secret for the OpenAI API.
            websocket_base_url: The websocket base URL for the OpenAI API.
            timeout: The timeout for the OpenAI API.
            max_retries: The max retries for the OpenAI API.
            default_headers: The default headers for the OpenAI API.
            default_query: The default query for the OpenAI API.
            http_client: The http_client for the OpenAI API.
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
        self.base_url = base_url
        self.organization = organization
        self.project = project
        self.webhook_secret = webhook_secret
        self.websocket_base_url = websocket_base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client

        self.memory_adapter = OpenAIMemoryAdapter()
        self._set_client()

    def _set_client(self):
        if not self.client:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                project=self.project,
                webhook_secret=self.webhook_secret,
                websocket_base_url=self.websocket_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                default_headers=self.default_headers,
                default_query=self.default_query,
                http_client=self.http_client,
            )

    def _set_a_client(self):
        if not self.a_client:
            self.a_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                project=self.project,
                webhook_secret=self.webhook_secret,
                websocket_base_url=self.websocket_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                default_headers=self.default_headers,
                default_query=self.default_query,
            )

    def _response_to_client_response(
        self, response, tool_map: dict[str, Tool] | None
    ) -> ClientResponse:
        blocks = []

        # Handle new response format with direct content array
        if hasattr(response, "output_parsed"):
            blocks.append(StructuredBlock(content=response.output_parsed))

        if hasattr(response, "output") and response.output:
            for content_item in response.output:
                if isinstance(content_item, ResponseOutputMessage) and not isinstance(
                    content_item, ParsedResponseOutputMessage
                ):
                    for content in content_item.content:
                        if content.type == "output_text":
                            blocks.append(TextBlock(content=content.text))
                elif isinstance(content_item, ResponseReasoningItem):
                    if content_item.summary:
                        blocks.append(
                            ThoughtBlock(content=content_item.summary[0].text)
                        )

                elif isinstance(content_item, ResponseFunctionToolCall):
                    if not tool_map:
                        raise ValueError("Tool map is required")

                    tool = tool_map.get(content_item.name)
                    if not tool:
                        raise ValueError(f"Tool {content_item.name} not found")
                    blocks.append(
                        FunctionCallBlock(
                            id=content_item.call_id,
                            name=content_item.name,
                            arguments=json.loads(content_item.arguments)
                            if isinstance(content_item.arguments, str)
                            else content_item.arguments,
                            tool=tool,
                        )
                    )

        # Handle usage from new format
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "input_tokens", 0)
            completion_tokens = getattr(usage, "output_tokens", 0)
            cached_tokens = 0
            # Handle input_tokens_details for cached tokens
            if hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
                cached_tokens = getattr(usage.input_tokens_details, "cached_tokens", 0)

        # Handle stop reason - use status from new format
        stop_reason = getattr(response, "status", None)
        if not stop_reason and hasattr(response, "choices") and response.choices:
            stop_reason = response.choices[0].finish_reason

        return ClientResponse(
            content=blocks,
            stop_reason=stop_reason,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens or 0,
                completion_tokens=completion_tokens or 0,
                cached_tokens=cached_tokens or 0,
            ),
        )

    def _convert_tools(self, tool: Tool) -> dict:
        """Convert tools to OpenAI function format"""
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.properties,
                "required": tool.required,
            },
        }

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
                "name": tool_choice[0],
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
            **kwargs,
            "model": self.model_name,
            "input": messages,
            "stream": False,
            "max_output_tokens": max_tokens,
        }
        if temperature:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        client: OpenAI = self._get_client()
        response = client.responses.create(**kwargs)
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
            **kwargs,
            "model": self.model_name,
            "input": messages,
            "stream": False,
            "max_output_tokens": max_tokens,
        }
        if temperature:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        a_client = self._get_a_client()
        response = await a_client.responses.create(**kwargs)
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

        tool_map = {tool.name: tool for tool in tools}

        kwargs = {
            **kwargs,
            "model": self.model_name,
            "input": messages,
            "stream": True,
            "max_output_tokens": max_tokens,
            # "stream_options": {"include_usage": True},
        }
        if temperature:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        response = self.client.responses.create(**kwargs)
        for chunk in response:
            if isinstance(chunk, ResponseTextDeltaEvent):
                yield ClientResponse(
                    content=[],
                    delta=chunk.delta,
                    stop_reason=None,
                    usage=TokenUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        cached_tokens=0,
                    ),
                )

            if isinstance(chunk, ResponseCompletedEvent):
                yield self._response_to_client_response(chunk.response, tool_map)

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

        tool_map = {tool.name: tool for tool in tools}
        kwargs = {
            **kwargs,
            "model": self.model_name,
            "input": messages,
            "stream": True,
            "max_output_tokens": max_tokens,
            # "stream_options": {"include_usage": True},
        }
        if temperature:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        a_client = self._get_a_client()

        async for chunk in await a_client.responses.create(**kwargs):
            if isinstance(chunk, ResponseTextDeltaEvent):
                yield ClientResponse(
                    content=[],
                    delta=chunk.delta,
                    stop_reason=None,
                    usage=TokenUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        cached_tokens=0,
                    ),
                )

            if isinstance(chunk, ResponseCompletedEvent):
                yield self._response_to_client_response(chunk.response, tool_map)

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

        if tools is None:
            tools = []

        messages = self._memory_to_contents(system_prompt, input, memory)

        tool_map = {tool.name: tool for tool in tools}
        kwargs = {
            "model": self.model_name,
            "input": messages,
            "text_format": output_cls,
            "max_output_tokens": max_tokens,
            **kwargs,
        }
        if temperature:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)
            # Structured response needs strict mode and no additional properties
            for tool in kwargs["tools"]:
                tool["strict"] = True
                tool["parameters"]["additionalProperties"] = False

        response = self.client.responses.parse(**kwargs)

        return self._response_to_client_response(response, tool_map)

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
        if tools is None:
            tools = []

        messages = self._memory_to_contents(system_prompt, input, memory)
        tool_map = {tool.name: tool for tool in tools}

        kwargs = {
            "model": self.model_name,
            "input": messages,
            "text_format": output_cls,
            "max_output_tokens": max_tokens,
            **kwargs,
        }
        if temperature:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)
            # Structured response needs strict mode and no additional properties
            for tool in kwargs["tools"]:
                tool["strict"] = True
                tool["parameters"]["additionalProperties"] = False

        a_client = self._get_a_client()
        response = await a_client.responses.parse(**kwargs)

        return self._response_to_client_response(response, tool_map)

    def _embed(
        self, text: str | list[str], model_name: str | None, **kwargs
    ) -> list[float] | list[list[float]]:
        """Embed a text using the model"""
        response = self.client.embeddings.create(
            input=text, model=model_name or self.model_name, **kwargs
        )

        embeddings = [item.embedding for item in response.data]

        if isinstance(text, str):
            return embeddings[0] if embeddings else []

        return embeddings or []

    async def _a_embed(
        self, text: str | list[str], model_name: str | None, **kwargs
    ) -> list[float] | list[list[float]]:
        """Embed a text using the model"""

        a_client = self._get_a_client()
        response = await a_client.embeddings.create(
            input=text, model=model_name or self.model_name, **kwargs
        )

        embeddings = [item.embedding for item in response.data]

        if isinstance(text, str):
            return embeddings[0] if embeddings else []

        return embeddings or []

    def _is_azure_client(self) -> bool:
        return isinstance(self._get_client(), AzureOpenAI)
