import json
import logging
import os
import time
from abc import abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

from pydantic import BaseModel

from datapizza.core.cache import Cache, cacheable
from datapizza.core.clients.models import ClientResponse
from datapizza.core.models import ChainableProducer, PipelineComponent
from datapizza.memory import Memory
from datapizza.memory.memory_adapter import MemoryAdapter
from datapizza.tools import Tool
from datapizza.tracing.tracing import generation_span
from datapizza.type import Block, Model, TextBlock

# Use the module name to ensure proper hierarchical relationship with parent logger
log = logging.getLogger(__name__)


# Abstract class for all clients
class Client(ChainableProducer):
    """
    Represents the base class for all clients.
    Concrete implementations must implement the abstract methods to handle the actual inference.

    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        temperature: float | None = None,
        cache: Cache | None = None,
    ):
        log.debug(
            f"Initializing client with model_name: {model_name}, system_prompt: {system_prompt}, temperature: {temperature}"
        )
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.cache = cache
        self.client = None
        self.a_client = None
        self.memory_adapter: MemoryAdapter

    def _get_client(self) -> Any:
        if not self.client:
            self._set_client()
        return self.client

    def _set_client(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    def _get_a_client(self) -> Any:
        if not self.a_client:
            self._set_a_client()
        return self.a_client

    def _set_a_client(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    def _get_cache_key(self, args: dict) -> str:
        user_input = args.get("input", "")
        output_cls = args.get("output_cls")
        input_key: str = ""
        if isinstance(user_input, list):
            for b in user_input:
                input_key += str(hash(b))

        if isinstance(user_input, str):
            input_key = user_input

        key: str = (
            str(input_key)
            + str(self.model_name)
            + (args.get("system_prompt") or self.system_prompt or "")
            + (str(hash(args.get("memory"))) if args.get("memory") else "")
            + (str(output_cls.model_fields) if output_cls else "")
        )
        return key

    @cacheable(_get_cache_key)
    def invoke(
        self,
        input: str | list[Block],
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> ClientResponse:
        """
        Performs a single inference request to the model.

        Args:
            input (str): The input text/prompt to send to the model
            tools (List[Tool], optional): List of tools available for the model to use. Defaults to [].
            memory (Memory, optional): Memory object containing conversation history. Defaults to None.
            tool_choice (str, optional): Controls which tool to use. Defaults to "auto".
            temperature (float, optional): Controls randomness in responses. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens in the response. Defaults to None.
            system_prompt (str, optional): System-level instructions for the model. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the model's inference method

        Returns:
            A ClientResponse object containing the model's response
        """
        if tools is None:
            tools = []

        if isinstance(input, str) and input:
            input = [TextBlock(content=input)]
        elif input == "":
            input = []

        log.debug(
            f"Invoking model {self.__class__.__name__} with input: {str(input)[:100]}..."
        )

        with generation_span("client.invoke") as span:
            if tools is None:
                tools = []

            response = self._invoke(
                input=input,
                tools=tools,
                memory=memory,
                tool_choice=tool_choice,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt or self.system_prompt,
                **kwargs,
            )
            span.set_attribute("prompt_tokens_used", response.prompt_tokens_used)
            span.set_attribute(
                "completion_tokens_used", response.completion_tokens_used
            )
            span.set_attribute("cached_tokens_used", response.cached_tokens_used)
            span.set_attribute("model_name", self.model_name)
            if os.getenv("DATAPIZZA_TRACE_CLIENT_IO", "false").lower() == "true":
                span.set_attribute(
                    "input", json.dumps([b.to_dict() for b in input]) if input else ""
                )
                span.set_attribute("output", response.text)
                span.set_attribute("memory", memory.json_dumps() if memory else "None")
            span.set_attribute("stop_reason", response.stop_reason or "None")

        assert isinstance(response, ClientResponse)
        return response

    @cacheable(_get_cache_key)
    async def a_invoke(
        self,
        input: str | list[Block],
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> ClientResponse:
        """
        Performs a single inference request to the model.

        Args:
            input (str): The input text/prompt to send to the model
            tools (List[Tool], optional): List of tools available for the model to use. Defaults to [].
            memory (Memory, optional): Memory object containing conversation history. Defaults to None.
            tool_choice (str, optional): Controls which tool to use. Defaults to "auto".
            temperature (float, optional): Controls randomness in responses. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens in the response. Defaults to None.
            system_prompt (str, optional): System-level instructions for the model. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the model's inference method

        Returns:
            A ClientResponse object containing the model's response
        """
        if isinstance(input, str) and input:
            input = [TextBlock(content=input)]
        elif input == "":
            input = []

        with generation_span("client.a_invoke") as span:
            if tools is None:
                tools = []
            log.debug(f"Invoking model {self.__class__.__name__} with input: {input}")

            if os.getenv("DATAPIZZA_TRACE_CLIENT_IO", "false").lower() == "true":
                span.set_attribute(
                    "input", json.dumps([b.to_dict() for b in input]) if input else ""
                )

            response = await self._a_invoke(
                input=input,
                tools=tools,
                memory=memory,
                tool_choice=tool_choice,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt or self.system_prompt,
                **kwargs,
            )
            if os.getenv("DATAPIZZA_TRACE_CLIENT_IO", "false").lower() == "true":
                span.set_attribute("output", response.text)
                span.set_attribute("memory", memory.json_dumps() if memory else "None")
            span.set_attribute("prompt_tokens_used", response.prompt_tokens_used)
            span.set_attribute(
                "completion_tokens_used", response.completion_tokens_used
            )
            span.set_attribute("cached_tokens_used", response.cached_tokens_used)
            span.set_attribute("model_name", self.model_name)
            span.set_attribute("stop_reason", response.stop_reason or "None")

        assert isinstance(response, ClientResponse)
        return response

    def stream_invoke(
        self,
        input: str | list[Block],
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> Iterator[ClientResponse]:
        """
        Streams the model's response token by token.

        Args:
            input (str): The input text/prompt to send to the model
            tools (List[Tool], optional): List of tools available for the model to use. Defaults to [].
            memory (Memory, optional): Memory object containing conversation history. Defaults to None.
            tool_choice (str, optional): Controls which tool to use. Defaults to "auto".
            temperature (float, optional): Controls randomness in responses. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens in the response. Defaults to None.
            system_prompt (str, optional): System-level instructions for the model. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the model's inference method

        Returns:
            An iterator yielding ClientResponse objects containing the model's response
        """

        if isinstance(input, str) and input:
            input = [TextBlock(content=input)]
        elif input == "":
            input = []

        with generation_span("client.stream_invoke") as span:
            if tools is None:
                tools = []
            log.debug(f"Streaming invoke model {self.__class__.__name__}")
            first_response = True

            response_stream = self._stream_invoke(
                input,
                tools,
                memory,
                tool_choice,
                temperature or self.temperature,
                max_tokens,
                system_prompt or self.system_prompt,
                **kwargs,
            )
            last_response = None
            for r in response_stream:
                if first_response:
                    span.set_attribute("time_at_first_token", int(time.time() * 1000))
                    first_response = False

                last_response = r
                yield r

            if last_response:
                span.set_attribute(
                    "prompt_tokens_used", last_response.prompt_tokens_used
                )
                span.set_attribute(
                    "completion_tokens_used", last_response.completion_tokens_used
                )
                span.set_attribute(
                    "cached_tokens_used", last_response.cached_tokens_used
                )
                span.set_attribute("model_name", self.model_name)
                span.set_attribute("stop_reason", last_response.stop_reason or "None")
                if os.getenv("DATAPIZZA_TRACE_CLIENT_IO", "false").lower() == "true":
                    span.set_attribute(
                        "input",
                        json.dumps([b.to_dict() for b in input]) if input else "",
                    )
                    span.set_attribute("output", last_response.text)
                    span.set_attribute(
                        "memory", memory.json_dumps() if memory else "None"
                    )

    async def a_stream_invoke(
        self,
        input: str | list[Block],
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[ClientResponse]:
        """
        Streams the model's response token by token asynchronously.

        Args:
            input (str): The input text/prompt to send to the model
            tools (List[Tool], optional): List of tools available for the model to use. Defaults to [].
            memory (Memory, optional): Memory object containing conversation history. Defaults to None.
            tool_choice (str, optional): Controls which tool to use. Defaults to "auto".
            temperature (float, optional): Controls randomness in responses. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens in the response. Defaults to None.
            system_prompt (str, optional): System-level instructions for the model. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the model's inference method

        Returns:
            An async iterator yielding ClientResponse objects containing the model's response
        """
        if isinstance(input, str) and input:
            input = [TextBlock(content=input)]
        elif input == "":
            input = []

        with generation_span("client.a_stream_invoke") as span:
            if tools is None:
                tools = []
            log.debug(f"Streaming invoke model {self.__class__.__name__}")
            first_response = True
            last_response = None

            if (
                input
                and os.getenv("DATAPIZZA_TRACE_CLIENT_IO", "false").lower() == "true"
            ):
                span.set_attribute(
                    "input",
                    json.dumps([b.to_dict() for b in input]) if input else "",
                )

            response_stream = self._a_stream_invoke(
                input,
                tools,
                memory,
                tool_choice,
                temperature or self.temperature,
                max_tokens,
                system_prompt or self.system_prompt,
                **kwargs,
            )
            async for r in response_stream:  # type: ignore
                if first_response:
                    span.set_attribute("time_at_first_token", int(time.time() * 1000))
                    first_response = False

                last_response = r
                yield r

            if last_response:
                span.set_attribute(
                    "prompt_tokens_used", last_response.prompt_tokens_used
                )
                span.set_attribute(
                    "completion_tokens_used", last_response.completion_tokens_used
                )
                span.set_attribute(
                    "cached_tokens_used", last_response.cached_tokens_used
                )
                span.set_attribute("model_name", self.model_name)
                span.set_attribute("stop_reason", last_response.stop_reason or "None")

                if os.getenv("DATAPIZZA_TRACE_CLIENT_IO", "false").lower() == "true":
                    span.set_attribute("output", last_response.text)
                    span.set_attribute(
                        "memory", memory.json_dumps() if memory else "None"
                    )

    @cacheable(_get_cache_key)
    def structured_response(
        self,
        *,
        input: str | list[Block],
        output_cls: type[Model],
        memory: Memory | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        **kwargs,
    ) -> ClientResponse:
        """
        Structures the model's response according to a specified output class.

        Args:
            input (str): The input text/prompt to send to the model
            output_cls (Type[Model]): The class type to structure the response into
            memory (Memory, optional): Memory object containing conversation history. Defaults to None.
            temperature (float, optional): Controls randomness in responses. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens in the response. Defaults to None.
            system_prompt (str, optional): System-level instructions for the model. Defaults to None.
            tools (List[Tool], optional): List of tools available for the model to use. Defaults to [].
            tool_choice (Literal["auto", "required", "none"] | list[str], optional): Controls which tool to use ("auto" by default). Defaults to "auto".
            **kwargs: Additional keyword arguments to pass to the model's inference method

        Returns:
            A ClientResponse object containing the structured response
        """
        if output_cls != {"type": "json_object"} and not issubclass(
            output_cls, BaseModel
        ):
            raise ValueError(
                "output_cls must be a subclass of Model or must be {'type': 'json_object'}"
            )

        if isinstance(input, str) and input:
            input = [TextBlock(content=input)]
        elif input == "":
            input = []

        with generation_span("client.structured_response") as span:
            log.debug(
                f"Structured response model {self.__class__.__name__} with input: {input}, memory: {memory}"
            )

            response = self._structured_response(
                input,
                output_cls,
                memory,
                temperature or self.temperature,
                max_tokens,
                system_prompt or self.system_prompt,
                tools,
                tool_choice,
                **kwargs,
            )

            span.set_attribute("prompt_tokens_used", response.prompt_tokens_used)
            span.set_attribute(
                "completion_tokens_used", response.completion_tokens_used
            )
            span.set_attribute("cached_tokens_used", response.cached_tokens_used)
            span.set_attribute("model_name", self.model_name)
            span.set_attribute("stop_reason", response.stop_reason or "None")
            if os.getenv("DATAPIZZA_TRACE_CLIENT_IO", "false").lower() == "true":
                span.set_attribute(
                    "input", json.dumps([b.to_dict() for b in input]) if input else ""
                )
                span.set_attribute("output", str(response.structured_data[0]))
                span.set_attribute("memory", memory.json_dumps() if memory else "None")

        assert isinstance(response, ClientResponse)
        return response

    @cacheable(_get_cache_key)
    async def a_structured_response(
        self,
        *,
        input: str | list[Block],
        output_cls: type[Model],
        memory: Memory | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        **kwargs,
    ) -> ClientResponse:
        """
        Structures the model's response according to a specified output class.

        Args:
            input (str): The input text/prompt to send to the model
            output_cls (Type[Model]): The class type to structure the response into
            memory (Memory, optional): Memory object containing conversation history. Defaults to None.
            temperature (float, optional): Controls randomness in responses. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens in the response. Defaults to None.
            system_prompt (str, optional): System-level instructions for the model. Defaults to None.
            tools (List[Tool], optional): List of tools available for the model to use. Defaults to [].
            tool_choice (Literal["auto", "required", "none"] | list[str], optional): Controls which tool to use ("auto" by default). Defaults to "auto".
            **kwargs: Additional keyword arguments to pass to the model's inference method

        Returns:
            A ClientResponse object containing the structured response
        """
        if isinstance(input, str) and input:
            input = [TextBlock(content=input)]
        elif input == "":
            input = []

        with generation_span("client.a_structured_response") as span:
            if output_cls != {"type": "json_object"} and not issubclass(
                output_cls, BaseModel
            ):
                raise ValueError(
                    "output_cls must be a subclass of Model or must be {'type': 'json_object'}"
                )

            log.debug(
                f"Structured response model {self.__class__.__name__} with input: {input}, memory: {memory}"
            )

            response = await self._a_structured_response(
                input,
                output_cls,
                memory,
                temperature or self.temperature,
                max_tokens,
                system_prompt or self.system_prompt,
                tools,
                tool_choice,
                **kwargs,
            )
            span.set_attribute("prompt_tokens_used", response.prompt_tokens_used)
            span.set_attribute(
                "completion_tokens_used", response.completion_tokens_used
            )
            span.set_attribute("cached_tokens_used", response.cached_tokens_used)
            span.set_attribute("model_name", self.model_name)
            span.set_attribute("stop_reason", response.stop_reason or "None")
            if os.getenv("DATAPIZZA_TRACE_CLIENT_IO", "false").lower() == "true":
                span.set_attribute(
                    "input", json.dumps([b.to_dict() for b in input]) if input else ""
                )
                span.set_attribute("output", str(response.structured_data[0]))
                span.set_attribute("memory", memory.json_dumps() if memory else "None")

        assert isinstance(response, ClientResponse)
        return response

    @abstractmethod
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
        """
        Internal method to implement the actual inference call.
        Must be implemented by concrete classes.
        """

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        """
        Internal method to implement the streaming inference call.
        Must be implemented by concrete classes.
        """
        pass

    @abstractmethod
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
        """
        Internal method to implement the async streaming inference call.
        Must be implemented by concrete classes.
        """
        pass

    @abstractmethod
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
        """
        Internal method to implement the structured response call.
        Must be implemented by concrete classes.
        """
        pass

    def _memory_to_contents(
        self, system_prompt: str | None, input: str, memory: Memory | None = None
    ) -> list[dict]:
        """Convert memory and input into GenAI content format"""
        contents = []
        contents.extend(
            self.memory_adapter.memory_to_messages(memory, system_prompt, input)
        )

        return contents

    @cacheable(
        lambda self, args: "|".join(args.get("text"))
        + "|"
        + args.get("model_name", self.model_name)
        + "|embed"
    )
    def embed(
        self, text: str | list[str], model_name: str | None = None, **kwargs
    ) -> list[float]:
        """Embed a text using the model

        Args:
            text (str | list[str]): The text to embed
            model_name (str, optional): The name of the model to use. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the model's embedding method

        Returns:
            list[float]: The embedding vector for the text
        """
        return self._embed(text, model_name, **kwargs)

    async def a_embed(
        self, text: str | list[str], model_name: str | None = None, **kwargs
    ) -> list[float] | list[list[float]]:
        """Embed a text using the model

        Args:
            text (str | list[str]): The text to embed
            model_name (str, optional): The name of the model to use. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the model's embedding method

        Returns:
            list[float]: The embedding vector for the text
        """
        return await self._a_embed(text, model_name, **kwargs)

    def _embed(
        self, text: str | list[str], model_name: str | None = None, **kwargs
    ) -> list[float]:
        """Embed a text using the model"""
        raise NotImplementedError("Embedding is not implemented for this client")

    async def _a_embed(
        self, text: str | list[str], model_name: str | None = None, **kwargs
    ) -> list[float] | list[list[float]]:
        """Embed a text using the model"""
        raise NotImplementedError("async Embedding is not implemented for this client")

    @abstractmethod
    def _convert_tool_choice(self, tool_choice: str | list[str]) -> dict:
        """Convert tool choice to the client's format"""
        pass

    def _as_module_component(self):
        return InferenceClientModule(self)


class InferenceClientModule(PipelineComponent):
    def __init__(self, client: Client):
        self.client = client

    def _run(self, **kwargs):
        return self.client.invoke(**kwargs)

    async def _a_run(self, **kwargs):
        return await self.client.a_invoke(**kwargs)


class StreamInferenceClientModule(PipelineComponent):
    def __init__(self, client: Client):
        self.client = client

    def _run(self, **kwargs):
        return self.client.stream_invoke(**kwargs)

    async def _a_run(self, **kwargs):
        return self.client.a_stream_invoke(**kwargs)
