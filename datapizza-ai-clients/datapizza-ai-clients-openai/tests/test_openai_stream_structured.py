import asyncio
import json
from types import SimpleNamespace

from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent
from pydantic import BaseModel

from datapizza.clients.openai.memory_adapter import OpenAIMemoryAdapter
from datapizza.clients.openai.openai_client import OpenAIClient


class Person(BaseModel):
    name: str
    age: int


def _delta_event(delta):
    return ResponseTextDeltaEvent.model_construct(
        type="response.output_text.delta",
        delta=delta,
    )


def _completed_event(*, output_parsed, output_text=None):
    response = SimpleNamespace(
        output_parsed=output_parsed,
        output_text=output_text,
        output=[],
        status="completed",
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=9,
            input_tokens_details=None,
            output_tokens_details=None,
        ),
    )
    return ResponseCompletedEvent.model_construct(
        type="response.completed",
        response=response,
    )


class FakeStreamManager:
    def __init__(self, events):
        self._events = events

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, *args):
        return False


class FakeAsyncStreamManager:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        async def iterator():
            for event in self._events:
                yield event

        return iterator()

    async def __aexit__(self, *args):
        return False


class FakeResponses:
    def __init__(self, events, *, is_async=False):
        self._events = events
        self._is_async = is_async
        self.captured_kwargs = None

    def stream(self, **kwargs):
        self.captured_kwargs = kwargs
        if self._is_async:
            return FakeAsyncStreamManager(self._events)
        return FakeStreamManager(self._events)


def _make_client(events, *, sync=True):
    client = object.__new__(OpenAIClient)
    client.model_name = "gpt-test"
    client.system_prompt = ""
    client.temperature = None
    client.memory_adapter = OpenAIMemoryAdapter()
    fake = FakeResponses(events, is_async=not sync)
    if sync:
        client.client = SimpleNamespace(responses=fake)
    else:
        client.a_client = SimpleNamespace(responses=fake)
    return client, fake


def _structured_events():
    return [
        _delta_event('{"name":"Ada"'),
        _delta_event(',"age":36}'),
        _completed_event(output_parsed=Person(name="Ada", age=36)),
    ]


def test_stream_structured_response():
    client, fake = _make_client(_structured_events())

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
    assert final_response.stop_reason == "completed"
    assert final_response.usage.prompt_tokens == 5
    assert final_response.usage.completion_tokens == 9

    assert fake.captured_kwargs["text_format"] is Person
    assert fake.captured_kwargs["model"] == "gpt-test"


def test_a_stream_structured_response():
    client, fake = _make_client(_structured_events(), sync=False)

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
    assert final_response.usage.completion_tokens == 9
    assert fake.captured_kwargs["text_format"] is Person


def test_stream_structured_response_forces_strict_tools():
    from datapizza.tools import tool

    @tool
    def get_weather(location: str) -> str:
        return "25 C"

    client, fake = _make_client(_structured_events())

    list(
        client._stream_structured_response(
            input="Tell me about Ada",
            output_cls=Person,
            tools=[get_weather],
        )
    )

    converted_tool = fake.captured_kwargs["tools"][0]
    assert converted_tool["strict"] is True
    assert converted_tool["parameters"]["additionalProperties"] is False


def test_stream_structured_response_json_object_sentinel():
    events = [
        _delta_event('{"name":"Ada"}'),
        _completed_event(output_parsed=None, output_text='{"name":"Ada"}'),
    ]
    client, fake = _make_client(events)

    responses = list(
        client._stream_structured_response(
            input="Tell me about Ada",
            output_cls={"type": "json_object"},
        )
    )

    assert fake.captured_kwargs["text"] == {"format": {"type": "json_object"}}
    assert "text_format" not in fake.captured_kwargs
    assert responses[-1].structured_data == [{"name": "Ada"}]


def test_stream_structured_response_json_object_sentinel_invalid_json():
    events = [
        _delta_event('{"name": '),
        _completed_event(output_parsed=None, output_text='{"name": '),
    ]
    client, _ = _make_client(events)

    try:
        list(
            client._stream_structured_response(
                input="Tell me about Ada",
                output_cls={"type": "json_object"},
            )
        )
        raise AssertionError("expected json.JSONDecodeError")
    except json.JSONDecodeError:
        pass
