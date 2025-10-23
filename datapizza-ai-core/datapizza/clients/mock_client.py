import logging
from typing import Literal

from datapizza.core.clients import ClientResponse
from datapizza.core.clients.client import Client
from datapizza.core.clients.models import TokenUsage
from datapizza.memory.memory import Memory, Turn
from datapizza.memory.memory_adapter import MemoryAdapter
from datapizza.tools.tools import Tool
from datapizza.type import (
    ROLE,
    Block,
    FunctionCallBlock,
    FunctionCallResultBlock,
    Model,
    StructuredBlock,
    TextBlock,
)

log = logging.getLogger(__name__)


class FakeMemoryAdapter(MemoryAdapter):
    def _text_to_message(self, text: str, role: ROLE) -> dict:
        return {"role": role.value, "content": text}

    def _turn_to_message(self, turn: Turn) -> dict:
        return {"role": turn.role.value, "blocks": turn.blocks}


class MockClient(Client):
    """A client for interacting with the Mock API.

    This class provides methods for invoking the Mock API to generate responses
    based on given input data. It extends the InferenceClient class.
    """

    def __init__(
        self,
        model_name: str | None = None,
        system_prompt: str = "",
        temperature: float = 0.6,
    ):
        super().__init__(
            model_name or "mock_client",
            system_prompt=system_prompt,
            temperature=temperature,
        )

        self.memory_adapter = FakeMemoryAdapter()

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
    ):
        if tools is None:
            tools = []

        input_text = ""
        if isinstance(input, list):
            for b in input:
                if isinstance(b, TextBlock):
                    input_text = b.content

        if memory and isinstance(memory[-1].blocks[-1], FunctionCallResultBlock):
            return ClientResponse(
                content=[TextBlock(content=memory[-1].blocks[-1].result)]
            )

        if not input_text:
            return ClientResponse(
                content=[
                    TextBlock(
                        content="hi i got this input: "
                        + " and a memory of length: "
                        + (str(len(memory)) if memory else "None")
                    )
                ]
            )

        if "function" in input_text and tools:
            arguments = {
                "text": "This is a test",
            }
            return ClientResponse(
                content=[
                    FunctionCallBlock(
                        id="1",
                        arguments=arguments,
                        name=tools[0].name,
                        tool=tools[0],
                    )
                ]
            )

        if "exception" in input_text:
            raise Exception("This is a test exception")

        if memory:
            text = ""
            for b in memory.iter_blocks():
                text += b.content

            text += input_text

            return ClientResponse(
                content=[TextBlock(content=text)],
                usage=TokenUsage(
                    prompt_tokens=len(input_text),
                    completion_tokens=len(text),
                    cached_tokens=0,
                ),
            )

        return ClientResponse(
            content=[TextBlock(content=input_text)],
            usage=TokenUsage(
                prompt_tokens=len(input_text),
                completion_tokens=len(input_text),
                cached_tokens=0,
            ),
        )

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
    ):
        if tools is None:
            tools = []
        return self._invoke(
            input=input,
            tools=tools,
            memory=memory,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs,
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
        **kwargs,
    ):
        if isinstance(input[0], TextBlock):
            return ClientResponse(
                content=[
                    StructuredBlock(
                        content=output_cls.model_validate_json(input[0].content)
                    )
                ]
            )
        else:
            raise ValueError("input must be a list of TextBlock")

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
    ):
        if tools is None:
            tools = []
        given_response = ""

        if not isinstance(input[0], TextBlock):
            raise ValueError("input must be a list of TextBlock")

        for char in input[0].content:
            given_response += char
            yield ClientResponse(
                content=[TextBlock(content=given_response)], delta=char
            )

    async def _a_stream_invoke(  # type: ignore
        self,
        input: list[Block],
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
        if tools is None:
            tools = []
        given_response = ""
        for char in input[0].content:  # type: ignore
            given_response += char
            yield ClientResponse(
                content=[TextBlock(content=given_response)], delta=char
            )

    def _convert_tool_choice(self, tool_choice: str | list[str]) -> dict:
        return {}

    def _a_structured_response(  # type: ignore
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
    ):
        return self._structured_response(
            input=input,
            output_cls=output_cls,
            memory=memory,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
