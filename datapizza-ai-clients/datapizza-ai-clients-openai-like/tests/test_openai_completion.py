import asyncio
from types import SimpleNamespace

from datapizza.tools import tool
from datapizza.type import FunctionCallBlock

from datapizza.clients.openai_like import OpenAILikeClient


def _chunk(*, content=None, finish_reason=None, tool_calls=None, usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def _usage_chunk():
    return SimpleNamespace(choices=[], usage=None)


class FakeStreamClient:
    def __init__(self, chunks):
        self._chunks = chunks
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: iter(self._chunks))
        )


class FakeAsyncStreamClient:
    def __init__(self, chunks):
        self._chunks = chunks
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        async def iterator():
            for chunk in self._chunks:
                yield chunk

        return iterator()


@tool
def get_weather(location: str, when: str) -> str:
    return "25 C"


def test_init():
    client = OpenAILikeClient(
        api_key="test_api_key",
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant that can answer questions about piadina only in italian.",
    )
    assert client is not None


def test_stream_invoke_reconstructs_tool_calls():
    client = OpenAILikeClient(api_key="test", model="test-model")
    client.client = FakeStreamClient(
        [
            _chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id="call_1",
                        function=SimpleNamespace(
                            name="get_weather",
                            arguments='{"location":"Mi',
                        ),
                    )
                ]
            ),
            _chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id=None,
                        function=SimpleNamespace(
                            name=None,
                            arguments='lan","when":"tomorrow"}',
                        ),
                    )
                ]
            ),
            _chunk(finish_reason="tool_calls"),
        ]
    )

    responses = list(
        client.stream_invoke(
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
    function_calls = final_response.function_calls
    assert len(function_calls) == 1
    assert isinstance(function_calls[0], FunctionCallBlock)
    assert function_calls[0].name == "get_weather"
    assert function_calls[0].arguments == {
        "location": "Milan",
        "when": "tomorrow",
    }
    assert final_response.stop_reason == "tool_calls"


def test_stream_invoke_ignores_usage_only_chunks_without_reusing_stale_delta():
    client = OpenAILikeClient(api_key="test", model="test-model")
    client.client = FakeStreamClient(
        [
            _chunk(content="Hel"),
            _usage_chunk(),
            _chunk(content="lo", finish_reason="stop"),
        ]
    )

    responses = list(
        client.stream_invoke(
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


def test_a_stream_invoke_reconstructs_tool_calls():
    client = OpenAILikeClient(api_key="test", model="test-model")
    client.a_client = FakeAsyncStreamClient(
        [
            _chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id="call_1",
                        function=SimpleNamespace(
                            name="get_weather",
                            arguments='{"location":"Mi',
                        ),
                    )
                ]
            ),
            _chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id=None,
                        function=SimpleNamespace(
                            name=None,
                            arguments='lan","when":"tomorrow"}',
                        ),
                    )
                ]
            ),
            _chunk(finish_reason="tool_calls"),
        ]
    )

    async def collect():
        items = []
        async for response in client.a_stream_invoke(
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
    function_calls = final_response.function_calls
    assert len(function_calls) == 1
    assert isinstance(function_calls[0], FunctionCallBlock)
    assert function_calls[0].name == "get_weather"
    assert function_calls[0].arguments == {
        "location": "Milan",
        "when": "tomorrow",
    }
    assert final_response.stop_reason == "tool_calls"
