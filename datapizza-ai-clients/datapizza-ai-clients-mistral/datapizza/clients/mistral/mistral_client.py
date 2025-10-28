import json
import logging
import os
from collections.abc import AsyncIterator, Iterator
from typing import Literal

import requests
from datapizza.core.cache import Cache
from datapizza.core.clients import Client, ClientResponse
from datapizza.core.clients.models import TokenUsage
from datapizza.memory import Memory
from datapizza.tools import Tool
from datapizza.type import (
    FunctionCallBlock,
    Media,
    MediaBlock,
    Model,
    StructuredBlock,
    TextBlock,
)
from mistralai import Mistral
from mistralai.models.ocrresponse import OCRResponse

from .memory_adapter import MistralMemoryAdapter

log = logging.getLogger(__name__)


class MistralClient(Client):
    """A client for interacting with the Mistral API.

    This class provides methods for invoking the Mistral API to generate responses
    based on given input data. It extends the Client class.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-large-latest",
        system_prompt: str = "",
        temperature: float | None = None,
        cache: Cache | None = None,
    ):
        """
        Args:
            api_key: The API key for the Mistral API.
            model: The model to use for the Mistral API.
            system_prompt: The system prompt to use for the Mistral API.
            temperature: The temperature to use for the Mistral API.
            cache: The cache to use for the Mistral API.
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
        self.memory_adapter = MistralMemoryAdapter()
        self._set_client()

    def _set_client(self):
        self.client = Mistral(api_key=self.api_key)

    def _response_to_client_response(
        self, response, tool_map: dict[str, Tool] | None = None
    ) -> ClientResponse:
        blocks = []
        for choice in response.choices:
            if choice.message.content:
                blocks.append(TextBlock(content=choice.message.content))

            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    tool = tool_map.get(tool_call.function.name) if tool_map else None

                    if tool is None:
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

        log.debug(f"{self.__class__.__name__} response = {response}")
        return ClientResponse(
            content=blocks,
            stop_reason=response.choices[0].finish_reason,
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                cached_tokens=0,
            ),
        )

    def _convert_tools(self, tools: Tool) -> dict:
        """Convert tools to Mistral function format"""
        return {"type": "function", "function": tools.schema}

    def _convert_tool_choice(
        self, tool_choice: Literal["auto", "required", "none"] | list[str]
    ) -> dict | Literal["auto", "required", "none"]:
        if isinstance(tool_choice, list) and len(tool_choice) > 1:
            raise NotImplementedError(
                "multiple function names is not supported by Mistral"
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
        log.debug(f"{self.__class__.__name__} input = {input}")
        messages = self._memory_to_contents(system_prompt, input, memory)

        tool_map = {tool.name: tool for tool in tools}

        request_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            **kwargs,
        }

        if temperature:
            request_params["temperature"] = temperature

        if tools:
            request_params["tools"] = [self._convert_tools(tool) for tool in tools]
            request_params["tool_choice"] = self._convert_tool_choice(tool_choice)

        response = self.client.chat.complete(**request_params)
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
        log.debug(f"{self.__class__.__name__} input = {input}")
        messages = self._memory_to_contents(system_prompt, input, memory)

        tool_map = {tool.name: tool for tool in tools}

        request_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            **kwargs,
        }

        if temperature:
            request_params["temperature"] = temperature

        if tools:
            request_params["tools"] = [self._convert_tools(tool) for tool in tools]
            request_params["tool_choice"] = self._convert_tool_choice(tool_choice)

        response = await self.client.chat.complete_async(**request_params)
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
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs,
        }

        if temperature:
            request_params["temperature"] = temperature

        if tools:
            request_params["tools"] = [self._convert_tools(tool) for tool in tools]
            request_params["tool_choice"] = self._convert_tool_choice(tool_choice)

        response = self.client.chat.stream(**request_params)
        text = ""
        usage = TokenUsage()
        stop_reason = None
        for chunk in response:
            usage += TokenUsage(
                prompt_tokens=chunk.data.usage.prompt_tokens
                if chunk.data.usage
                else 0 or 0,
                completion_tokens=chunk.data.usage.completion_tokens
                if chunk.data.usage
                else 0 or 0,
                cached_tokens=0,
            )
            stop_reason = chunk.data.choices[0].finish_reason
            delta = chunk.data.choices[0].delta.content or ""
            text += delta
            yield ClientResponse(
                content=[],
                delta=str(delta),
                stop_reason=stop_reason,
            )

        yield ClientResponse(
            content=[TextBlock(content=text)],
            stop_reason=stop_reason,
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
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or 1024,
            **kwargs,
        }

        if temperature:
            request_params["temperature"] = temperature

        if tools:
            request_params["tools"] = [self._convert_tools(tool) for tool in tools]
            request_params["tool_choice"] = self._convert_tool_choice(tool_choice)

        response = await self.client.chat.stream_async(**request_params)
        text = ""
        usage = TokenUsage()
        stop_reason = None
        async for chunk in response:
            usage += TokenUsage(
                prompt_tokens=chunk.data.usage.prompt_tokens
                if chunk.data.usage
                else 0 or 0,
                completion_tokens=chunk.data.usage.completion_tokens
                if chunk.data.usage
                else 0 or 0,
                cached_tokens=0,
            )
            stop_reason = chunk.data.choices[0].finish_reason
            delta = chunk.data.choices[0].delta.content or ""
            text += delta
            yield ClientResponse(
                content=[],
                delta=str(delta),
                stop_reason=stop_reason,
            )

        yield ClientResponse(
            content=[TextBlock(content=text)],
            stop_reason=stop_reason,
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

        if not tools:
            tools = []

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        response = self.client.chat.parse(
            model=self.model_name,
            messages=messages,
            response_format=output_cls,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        if not response.choices:
            raise ValueError("No response from Mistral")

        log.debug(f"{self.__class__.__name__} structured response: {response}")
        stop_reason = response.choices[0].finish_reason if response.choices else None
        if hasattr(output_cls, "model_validate_json"):
            structured_data = output_cls.model_validate_json(
                str(response.choices[0].message.content)  # type: ignore
            )
        else:
            structured_data = json.loads(str(response.choices[0].message.content))  # type: ignore
        return ClientResponse(
            content=[StructuredBlock(content=structured_data)],
            stop_reason=stop_reason,
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                cached_tokens=0,
            ),
        )

    async def _a_structured_response(
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

        if not tools:
            tools = []

        if tools:
            kwargs["tools"] = [self._convert_tools(tool) for tool in tools]
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        response = await self.client.chat.parse_async(
            model=self.model_name,
            messages=messages,
            response_format=output_cls,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        if not response.choices:
            raise ValueError("No response from Mistral")

        log.debug(f"{self.__class__.__name__} structured response: {response}")
        stop_reason = response.choices[0].finish_reason if response.choices else None
        if hasattr(output_cls, "model_validate_json"):
            structured_data = output_cls.model_validate_json(
                str(response.choices[0].message.content)  # type: ignore
            )
        else:
            structured_data = json.loads(str(response.choices[0].message.content))  # type: ignore
        return ClientResponse(
            content=[StructuredBlock(content=structured_data)],
            stop_reason=stop_reason,
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                cached_tokens=0,
            ),
        )

    def _embed(
        self, text: str | list[str], model_name: str | None, **kwargs
    ) -> list[float] | list[list[float]]:
        """Embed a text using the model"""
        response = self.client.embeddings.create(
            inputs=text, model=model_name or self.model_name, **kwargs
        )

        embeddings = [item.embedding for item in response.data]

        if not embeddings:
            return []

        if isinstance(text, str) and embeddings[0]:
            return embeddings[0]

        return embeddings

    async def _a_embed(
        self, text: str | list[str], model_name: str | None, **kwargs
    ) -> list[float] | list[list[float]]:
        """Embed a text using the model"""
        response = await self.client.embeddings.create_async(
            inputs=text, model=model_name or self.model_name, **kwargs
        )

        embeddings = [item.embedding for item in response.data]

        if not embeddings:
            return []

        if isinstance(text, str) and embeddings[0]:
            return embeddings[0]

        return embeddings or []

    def parse_document(
        self,
        document_path: str,
        autodelete: bool = True,
        include_image_base64: bool = True,
    ) -> OCRResponse:
        filename = os.path.basename(document_path)
        with open(document_path, "rb") as f:
            uploaded_pdf = self.client.files.upload(
                file={"file_name": filename, "content": f}, purpose="ocr"
            )

        signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)

        response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            include_image_base64=include_image_base64,
        )

        if autodelete:
            url = f"https://api.mistral.ai/v1/files/{uploaded_pdf.id}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            requests.delete(url, headers=headers, timeout=30)

        return response
