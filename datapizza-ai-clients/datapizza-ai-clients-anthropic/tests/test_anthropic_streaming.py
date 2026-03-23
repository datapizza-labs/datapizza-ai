import asyncio
from types import SimpleNamespace

from datapizza.tools import tool
from datapizza.type import FunctionCallBlock

from datapizza.clients.anthropic.anthropic_client import AnthropicClient
from datapizza.clients.anthropic.memory_adapter import AnthropicMemoryAdapter


def _message_start(input_tokens=5):
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=input_tokens),
        ),
    )


def _message_delta(output_tokens=7, stop_reason="tool_use"):
    return SimpleNamespace(
        type="message_delta",
        usage=SimpleNamespace(output_tokens=output_tokens),
        delta=SimpleNamespace(stop_reason=stop_reason),
    )


def _tool_start(index, name, tool_id, input=None):
    return SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=SimpleNamespace(
            type="tool_use",
            id=tool_id,
            name=name,
            input=input,
        ),
    )


def _text_delta(text):
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(text=text),
    )


def _thinking_delta(thinking):
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(thinking=thinking),
    )


class FakeSyncMessages:
    def __init__(self, chunks):
        self._chunks = chunks

    def create(self, **kwargs):
        return iter(self._chunks)


class FakeAsyncMessages:
    def __init__(self, chunks):
        self._chunks = chunks

    async def create(self, **kwargs):
        async def iterator():
            for chunk in self._chunks:
                yield chunk

        return iterator()


class FakeSyncClient:
    def __init__(self, chunks):
        self.messages = FakeSyncMessages(chunks)


class FakeAsyncClient:
    def __init__(self, chunks):
        self.messages = FakeAsyncMessages(chunks)


def _make_client():
    client = object.__new__(AnthropicClient)
    client.model_name = "claude-test"
    client.system_prompt = ""
    client.temperature = None
    client.memory_adapter = AnthropicMemoryAdapter()
    return client


@tool
def get_weather(location: str, when: str) -> str:
    return "25 C"


def test_stream_invoke_preserves_tool_use_blocks():
    client = _make_client()
    client.client = FakeSyncClient(
        [
            _message_start(),
            _tool_start(
                index=0,
                name="get_weather",
                tool_id="tool_1",
                input={"location": "Milan", "when": "tomorrow"},
            ),
            _message_delta(stop_reason="tool_use"),
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
    assert final_response.stop_reason == "tool_use"


def test_stream_invoke_preserves_text_and_thinking():
    client = _make_client()
    client.client = FakeSyncClient(
        [
            _message_start(),
            _thinking_delta("Thinking..."),
            _text_delta("Hel"),
            _text_delta("lo"),
            _message_delta(stop_reason="end_turn"),
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
    assert responses[-1].thoughts == "Thinking..."


def test_a_stream_invoke_preserves_tool_use_blocks():
    client = _make_client()
    client.a_client = FakeAsyncClient(
        [
            _message_start(),
            _tool_start(
                index=0,
                name="get_weather",
                tool_id="tool_1",
                input={"location": "Milan", "when": "tomorrow"},
            ),
            _message_delta(stop_reason="tool_use"),
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
