import asyncio
from types import SimpleNamespace

from datapizza.tools import tool
from datapizza.type import FunctionCallBlock

from datapizza.clients.google.google_client import GoogleClient
from datapizza.clients.google.memory_adapter import GoogleMemoryAdapter


def _part(*, text=None, thought=False, function_call=None, thought_signature=None):
    return SimpleNamespace(
        text=text,
        thought=thought,
        function_call=function_call,
        thought_signature=thought_signature,
    )


def _chunk(parts, *, finish_reason="STOP", text=""):
    return SimpleNamespace(
        text=text,
        usage_metadata=None,
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

    def generate_content_stream(self, **kwargs):
        return iter(self._chunks)


class FakeAsyncModels:
    def __init__(self, chunks):
        self._chunks = chunks

    async def generate_content_stream(self, **kwargs):
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
