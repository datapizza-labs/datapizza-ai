import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
except ImportError:
    raise ImportError(
        "ibm-watsonx-ai is not installed. Please install it with `pip install ibm-watsonx-ai`"
    ) from None

from datapizza.core.cache import Cache
from datapizza.core.clients import Client, ClientResponse
from datapizza.core.clients.models import TokenUsage
from datapizza.memory import Memory
from datapizza.tools import Tool
from datapizza.type import Block, FunctionCallBlock, Model, TextBlock

from .memory_adapter import WatsonXMemoryAdapter

log = logging.getLogger(__name__)


class WatsonXClient(Client):
    """
    A client for interacting with the IBM WatsonX API.
    """

    def __init__(
        self,
        api_key: str,
        url: str,
        project_id: str,
        model: str = "ibm/granite-3-3-8b-instruct",
        system_prompt: str = "",
        temperature: float | None = None,
        cache: Cache | None = None,
    ):
        """
        Args:
            api_key: The API key for WatsonX.
            url: The endpoint URL for WatsonX.
            project_id: The project ID for WatsonX.
            model: The model ID to use.
            system_prompt: The system prompt to use.
            temperature: The temperature to use.
            cache: The cache to use.
        """
        super().__init__(
            model_name=model,
            system_prompt=system_prompt,
            temperature=temperature,
            cache=cache,
        )
        self.api_key = api_key
        self.url = url
        self.project_id = project_id

        self.memory_adapter = WatsonXMemoryAdapter()
        self._set_client()

    def _set_client(self):
        if not self.client:
            credentials = Credentials(url=self.url, api_key=self.api_key)
            api_client = APIClient(credentials, project_id=self.project_id)
            self.client = ModelInference(
                model_id=self.model_name, api_client=api_client
            )

    def _set_a_client(self):
        # Async client implementation if available in SDK, otherwise raise or wrap
        # Not strictly needed if we just use self.client.achat
        pass

    def _convert_tools(self, tool: Tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.properties,
                    "required": tool.required,
                },
            },
        }

    def _convert_tool_choice(
        self, tool_choice: Literal["auto", "required", "none"] | list[str]
    ) -> dict | str:
        if isinstance(tool_choice, list):
            if len(tool_choice) > 1:
                raise NotImplementedError("multiple function names is not supported")
            if tool_choice:
                return {"type": "function", "function": {"name": tool_choice[0]}}
            return "none"
        return tool_choice

    def _parse_response(
        self, response: dict, tool_map: dict[str, Tool] | None
    ) -> ClientResponse:
        """Helper method to parse the response from WatsonX API."""
        content_blocks: list[Block] = []
        usage = TokenUsage()
        stop_reason = None

        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            message = choice.get("message", {})
            text_content = message.get("content")
            if text_content:
                content_blocks.append(TextBlock(content=text_content))

            stop_reason = choice.get("finish_reason")

            tool_calls = message.get("tool_calls")
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call.get("type") == "function":
                        function = tool_call.get("function", {})
                        name = function.get("name")

                        arguments: dict[str, Any] = {}
                        raw_args = function.get("arguments")

                        if isinstance(raw_args, str):
                            try:
                                arguments = json.loads(raw_args)
                            except json.JSONDecodeError:
                                log.warning(
                                    f"Failed to parse arguments for tool {name}: {raw_args}"
                                )
                        elif isinstance(raw_args, dict):
                            arguments = raw_args

                        if tool_map and name in tool_map:
                            content_blocks.append(
                                FunctionCallBlock(
                                    id=tool_call.get("id"),
                                    name=name,
                                    arguments=arguments,
                                    tool=tool_map[name],
                                )
                            )

        if "usage" in response:
            resp_usage = response["usage"]
            usage = TokenUsage(
                prompt_tokens=resp_usage.get("prompt_tokens", 0),
                completion_tokens=resp_usage.get("completion_tokens", 0),
                cached_tokens=0,
            )

        return ClientResponse(
            content=content_blocks, stop_reason=stop_reason, usage=usage
        )

    def _invoke(
        self,
        input: list[Block],
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> ClientResponse:
        tool_map = {tool.name: tool for tool in tools} if tools else None

        # Prepare messages using memory adapter
        messages = self._memory_to_contents(
            system_prompt,
            input,
            memory,  # type: ignore
        )

        params = {}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        chat_kwargs = {}
        if tools:
            chat_kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            chat_kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)  # type: ignore

        # Invoke
        response = self.client.chat(
            messages=messages, params=params if params else None, **chat_kwargs
        )

        return self._parse_response(response, tool_map)

    async def _a_invoke(
        self,
        input: list[Block],
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> ClientResponse:
        tool_map = {tool.name: tool for tool in tools} if tools else None

        messages = self._memory_to_contents(
            system_prompt,
            input,
            memory,  # type: ignore
        )

        params = {}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        chat_kwargs = {}
        if tools:
            chat_kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            chat_kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)  # type: ignore

        # Invoke Async
        response = await self.client.achat(
            messages=messages, params=params if params else None, **chat_kwargs
        )

        return self._parse_response(response, tool_map)

    def _stream_invoke(
        self,
        input: list[Block],
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> Iterator[ClientResponse]:
        raise NotImplementedError("Stream invoke not implemented")

    async def _a_stream_invoke(
        self,
        input: list[Block],
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[ClientResponse]:
        raise NotImplementedError("Async stream invoke not implemented")

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
        **kwargs,
    ) -> ClientResponse:
        raise NotImplementedError("Structured response not implemented")

    async def _a_structured_response(
        self,
        input: list[Block] | None,
        output_cls: type[Model],
        memory: Memory | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        **kwargs,
    ) -> ClientResponse:
        raise NotImplementedError("Async structured response not implemented")
