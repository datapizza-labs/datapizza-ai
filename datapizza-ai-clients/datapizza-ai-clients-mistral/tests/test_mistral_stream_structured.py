import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from datapizza.clients.mistral import MistralClient


class Person(BaseModel):
    name: str
    age: int


def _chunk(*, content=None, finish_reason=None, usage=None):
    return SimpleNamespace(
        data=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )
    )


def _structured_chunks():
    return [
        _chunk(content='{"name":"Ada"'),
        _chunk(content=',"age":36}'),
        _chunk(
            finish_reason="stop",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=9, cached_tokens=0),
        ),
    ]


class FakeChat:
    def __init__(self, chunks):
        self._chunks = chunks
        self.captured_kwargs = None

    def stream(self, **kwargs):
        self.captured_kwargs = kwargs
        return iter(self._chunks)

    async def stream_async(self, **kwargs):
        self.captured_kwargs = kwargs

        async def iterator():
            for chunk in self._chunks:
                yield chunk

        return iterator()


def _make_client(chunks):
    client = MistralClient(api_key="test", model="test-model")
    fake_chat = FakeChat(chunks)
    client.client = SimpleNamespace(chat=fake_chat)
    return client, fake_chat


def test_stream_structured_response():
    client, fake_chat = _make_client(_structured_chunks())

    responses = list(
        client.stream_structured_response(input="Tell me about Ada", output_cls=Person)
    )

    deltas = responses[:-1]
    assert all(response.content == [] for response in deltas)
    assert "".join(response.delta for response in deltas) == '{"name":"Ada","age":36}'

    final_response = responses[-1]
    assert not final_response.delta
    assert final_response.structured_data == [Person(name="Ada", age=36)]
    assert final_response.stop_reason == "stop"
    assert final_response.usage.prompt_tokens == 5
    assert final_response.usage.completion_tokens == 9

    response_format = fake_chat.captured_kwargs["response_format"]
    assert response_format.type == "json_schema"
    assert response_format.json_schema.strict is True


def test_a_stream_structured_response():
    client, fake_chat = _make_client(_structured_chunks())

    async def collect():
        return [
            response
            async for response in client.a_stream_structured_response(
                input="Tell me about Ada", output_cls=Person
            )
        ]

    responses = asyncio.run(collect())

    deltas = responses[:-1]
    assert "".join(response.delta for response in deltas) == '{"name":"Ada","age":36}'

    final_response = responses[-1]
    assert not final_response.delta
    assert final_response.structured_data == [Person(name="Ada", age=36)]
    assert final_response.usage.completion_tokens == 9
    assert fake_chat.captured_kwargs["response_format"].type == "json_schema"


def test_stream_structured_response_invalid_json():
    client, _ = _make_client(
        [
            _chunk(content='{"name": '),
            _chunk(finish_reason="length"),
        ]
    )

    with pytest.raises(ValidationError):
        list(
            client.stream_structured_response(
                input="Tell me about Ada", output_cls=Person
            )
        )


def test_stream_structured_response_json_object_sentinel():
    client, fake_chat = _make_client(
        [
            _chunk(content='{"name":"Ada"}'),
            _chunk(finish_reason="stop"),
        ]
    )

    responses = list(
        client.stream_structured_response(
            input="Tell me about Ada", output_cls={"type": "json_object"}
        )
    )

    assert fake_chat.captured_kwargs["response_format"] == {"type": "json_object"}
    assert responses[-1].structured_data == [{"name": "Ada"}]
