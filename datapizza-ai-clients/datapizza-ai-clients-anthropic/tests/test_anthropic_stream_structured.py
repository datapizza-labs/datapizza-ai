import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from datapizza.clients.anthropic.anthropic_client import AnthropicClient
from datapizza.clients.anthropic.memory_adapter import AnthropicMemoryAdapter


class Person(BaseModel):
    name: str
    age: int


def _message_start(input_tokens=5):
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=input_tokens),
        ),
    )


def _message_delta(output_tokens=7, stop_reason="end_turn"):
    return SimpleNamespace(
        type="message_delta",
        usage=SimpleNamespace(output_tokens=output_tokens),
        delta=SimpleNamespace(stop_reason=stop_reason),
    )


def _text_delta(text):
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(text=text),
    )


class FakeSyncMessages:
    def __init__(self, chunks):
        self._chunks = chunks
        self.captured_kwargs = None

    def create(self, **kwargs):
        self.captured_kwargs = kwargs
        return iter(self._chunks)


class FakeAsyncMessages:
    def __init__(self, chunks):
        self._chunks = chunks
        self.captured_kwargs = None

    async def create(self, **kwargs):
        self.captured_kwargs = kwargs

        async def iterator():
            for chunk in self._chunks:
                yield chunk

        return iterator()


def _make_client(chunks, *, sync=True):
    client = object.__new__(AnthropicClient)
    client.model_name = "claude-test"
    client.system_prompt = ""
    client.temperature = None
    client.memory_adapter = AnthropicMemoryAdapter()
    if sync:
        client.client = SimpleNamespace(messages=FakeSyncMessages(chunks))
        return client, client.client.messages
    client.a_client = SimpleNamespace(messages=FakeAsyncMessages(chunks))
    return client, client.a_client.messages


def _structured_chunks():
    return [
        _message_start(input_tokens=5),
        _text_delta('{"name": "Ada"'),
        _text_delta(', "age": 36}'),
        _message_delta(output_tokens=7, stop_reason="end_turn"),
    ]


def test_stream_structured_response():
    client, fake_messages = _make_client(_structured_chunks())

    responses = list(
        client._stream_structured_response(
            input="Tell me about Ada",
            output_cls=Person,
        )
    )

    deltas = responses[:-1]
    assert all(response.content == [] for response in deltas)
    assert (
        "".join(response.delta for response in deltas) == '{"name": "Ada", "age": 36}'
    )

    final_response = responses[-1]
    assert not final_response.delta
    assert final_response.structured_data == [Person(name="Ada", age=36)]
    assert final_response.stop_reason == "end_turn"
    assert final_response.usage.prompt_tokens == 5
    assert final_response.usage.completion_tokens == 7

    assert fake_messages.captured_kwargs["stream"] is True
    output_format = fake_messages.captured_kwargs["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["properties"].keys() == {"name", "age"}


def test_a_stream_structured_response():
    client, fake_messages = _make_client(_structured_chunks(), sync=False)

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
    assert (
        "".join(response.delta for response in deltas) == '{"name": "Ada", "age": 36}'
    )

    final_response = responses[-1]
    assert not final_response.delta
    assert final_response.structured_data == [Person(name="Ada", age=36)]
    assert final_response.usage.completion_tokens == 7
    assert fake_messages.captured_kwargs["stream"] is True


def test_stream_structured_response_invalid_json():
    client, _ = _make_client(
        [
            _message_start(),
            _text_delta('{"name": '),
            _message_delta(stop_reason="max_tokens"),
        ]
    )

    with pytest.raises(ValidationError):
        list(
            client._stream_structured_response(
                input="Tell me about Ada",
                output_cls=Person,
            )
        )


def test_stream_structured_response_json_object_sentinel():
    client, fake_messages = _make_client(
        [
            _message_start(),
            _text_delta('{"name": "Ada"}'),
            _message_delta(stop_reason="end_turn"),
        ]
    )

    responses = list(
        client._stream_structured_response(
            input="Tell me about Ada",
            output_cls={"type": "json_object"},
        )
    )

    output_format = fake_messages.captured_kwargs["output_config"]["format"]
    assert output_format["schema"] == {"type": "object", "additionalProperties": True}
    assert responses[-1].structured_data == [{"name": "Ada"}]
