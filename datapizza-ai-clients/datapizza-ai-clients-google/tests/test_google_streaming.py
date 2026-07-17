import asyncio
from types import SimpleNamespace

import pytest
from datapizza.tools import tool
from datapizza.type import FunctionCallBlock
from pydantic import BaseModel, ValidationError

from datapizza.clients.google.google_client import GoogleClient
from datapizza.clients.google.memory_adapter import GoogleMemoryAdapter


def _part(*, text=None, thought=False, function_call=None, thought_signature=None):
    return SimpleNamespace(
        text=text,
        thought=thought,
        function_call=function_call,
        thought_signature=thought_signature,
    )


def _chunk(parts, *, finish_reason="STOP", text="", usage_metadata=None):
    return SimpleNamespace(
        text=text,
        usage_metadata=usage_metadata,
        candidates=[
            SimpleNamespace(
                finish_reason=SimpleNamespace(value=finish_reason),
                content=SimpleNamespace(parts=parts),
            )
        ],
    )


class FakeModels:
    def __init__(self, chunks):
        self._chunks = chunks
        self.captured_kwargs = None

    def generate_content_stream(self, **kwargs):
        self.captured_kwargs = kwargs
        return iter(self._chunks)


class FakeAsyncModels:
    def __init__(self, chunks):
        self._chunks = chunks
        self.captured_kwargs = None

    async def generate_content_stream(self, **kwargs):
        self.captured_kwargs = kwargs

        async def iterator():
            for chunk in self._chunks:
                yield chunk

        return iterator()


class FakeGoogleGenAIClient:
    def __init__(self, chunks):
        self.models = FakeModels(chunks)
        self.aio = SimpleNamespace(models=FakeAsyncModels(chunks))


def _make_client(chunks):
    client = object.__new__(GoogleClient)
    client.model_name = "gemini-test"
    client.system_prompt = ""
    client.temperature = None
    client.memory_adapter = GoogleMemoryAdapter()
    client.client = FakeGoogleGenAIClient(chunks)
    return client


@tool
def get_weather(location: str, when: str) -> str:
    return "25 C"


def test_stream_invoke_preserves_function_calls():
    client = _make_client(
        [
            _chunk(
                [
                    _part(
                        function_call=SimpleNamespace(
                            name="get_weather",
                            args={"location": "Milan", "when": "tomorrow"},
                        )
                    )
                ]
            )
        ]
    )

    responses = list(
        client._stream_invoke(
            input="weather?",
            tools=[get_weather],
            memory=None,
            tool_choice="auto",
            temperature=None,
            max_tokens=256,
            system_prompt=None,
        )
    )

    final_response = responses[-1]
    assert len(final_response.function_calls) == 1
    assert isinstance(final_response.function_calls[0], FunctionCallBlock)
    assert final_response.function_calls[0].name == "get_weather"
    assert final_response.function_calls[0].arguments == {
        "location": "Milan",
        "when": "tomorrow",
    }


def test_stream_invoke_uses_part_text_without_duplication():
    client = _make_client(
        [
            _chunk([_part(text="Hel")], text="Hel"),
            _chunk([_part(text="lo")], text="lo"),
        ]
    )

    responses = list(
        client._stream_invoke(
            input="hello",
            tools=[],
            memory=None,
            tool_choice="auto",
            temperature=None,
            max_tokens=256,
            system_prompt=None,
        )
    )

    assert [response.delta for response in responses[:-1]] == ["Hel", "lo"]
    assert responses[-1].text == "Hello"


def test_a_stream_invoke_preserves_function_calls():
    client = _make_client(
        [
            _chunk(
                [
                    _part(
                        function_call=SimpleNamespace(
                            name="get_weather",
                            args={"location": "Milan", "when": "tomorrow"},
                        )
                    )
                ]
            )
        ]
    )

    async def collect():
        items = []
        async for response in client._a_stream_invoke(
            input="weather?",
            tools=[get_weather],
            memory=None,
            tool_choice="auto",
            temperature=None,
            max_tokens=256,
            system_prompt=None,
        ):
            items.append(response)
        return items

    responses = asyncio.run(collect())
    final_response = responses[-1]
    assert len(final_response.function_calls) == 1
    assert final_response.function_calls[0].name == "get_weather"


class Person(BaseModel):
    name: str
    age: int


def _structured_chunks():
    return [
        _chunk([_part(text='{"name":"Ada"')], text='{"name":"Ada"'),
        _chunk(
            [_part(text=',"age":36}')],
            text=',"age":36}',
            usage_metadata=SimpleNamespace(
                prompt_token_count=4,
                candidates_token_count=11,
                cached_content_token_count=0,
                thoughts_token_count=0,
            ),
        ),
    ]


def test_stream_structured_response():
    client = _make_client(_structured_chunks())

    responses = list(
        client._stream_structured_response(
            input="Tell me about Ada",
            output_cls=Person,
        )
    )

    deltas = responses[:-1]
    assert all(response.content == [] for response in deltas)
    assert "".join(response.delta for response in deltas) == '{"name":"Ada","age":36}'

    final_response = responses[-1]
    assert not final_response.delta
    assert final_response.structured_data == [Person(name="Ada", age=36)]
    assert final_response.stop_reason == "stop"
    assert final_response.usage.prompt_tokens == 4
    assert final_response.usage.completion_tokens == 11

    config = client.client.models.captured_kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema == Person.model_json_schema()


def test_a_stream_structured_response():
    client = _make_client(_structured_chunks())

    async def collect():
        return [
            response
            async for response in client._a_stream_structured_response(
                input="Tell me about Ada",
                output_cls=Person,
            )
        ]

    responses = asyncio.run(collect())

    deltas = responses[:-1]
    assert "".join(response.delta for response in deltas) == '{"name":"Ada","age":36}'

    final_response = responses[-1]
    assert not final_response.delta
    assert final_response.structured_data == [Person(name="Ada", age=36)]
    assert final_response.usage.completion_tokens == 11

    config = client.client.aio.models.captured_kwargs["config"]
    assert config.response_mime_type == "application/json"


def test_stream_structured_response_tolerates_empty_chunks():
    empty_chunk = SimpleNamespace(text="", usage_metadata=None, candidates=[])
    client = _make_client(
        [
            _chunk([_part(text='{"name":"Ada"')], text='{"name":"Ada"'),
            empty_chunk,
            _chunk([_part(text=',"age":36}')], text=',"age":36}'),
        ]
    )

    responses = list(
        client._stream_structured_response(
            input="Tell me about Ada",
            output_cls=Person,
        )
    )

    assert responses[-1].structured_data == [Person(name="Ada", age=36)]


def test_stream_structured_response_invalid_json():
    client = _make_client(
        [
            _chunk([_part(text='{"name": ')], text='{"name": '),
        ]
    )

    with pytest.raises(ValidationError):
        list(
            client._stream_structured_response(
                input="Tell me about Ada",
                output_cls=Person,
            )
        )
