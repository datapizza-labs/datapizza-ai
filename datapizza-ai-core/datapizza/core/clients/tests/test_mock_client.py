import asyncio

import pytest
from pydantic import BaseModel, ValidationError

from datapizza.clients.mock_client import MockClient
from datapizza.core.clients import Client, ClientResponse
from datapizza.memory.memory import Memory
from datapizza.type import StructuredBlock, TextBlock
from datapizza.type.type import ROLE


def test_cannot_instantiate_abstract_class():
    with pytest.raises(TypeError):
        Client("a", "b", 0.5)


def test_mock_client():
    llm = MockClient(model_name="mock_client")

    response = ClientResponse(content=[TextBlock(content="Mock response")])

    assert llm.invoke(input="Mock response") == response


def test_stream_response():
    llm = MockClient(model_name="mock_client")
    text = "Here a long text to return"
    response = llm.stream_invoke(input=text)

    final_response = ""
    for i, res in enumerate(response):
        final_response += res.delta
        assert res.delta == text[i]
        assert res.text == text[: i + 1]

    assert final_response == text
    assert res.text == text


def test_memory():
    memory = Memory()
    textBlock = TextBlock(content="Mock response")
    textBlock2 = TextBlock(content="Mock response 2")
    memory.add_turn(textBlock, role=ROLE.ASSISTANT)
    assert memory[0].blocks == [textBlock]

    memory.clear()
    assert len(memory) == 0

    memory.add_turn([textBlock], role=ROLE.ASSISTANT)
    assert memory[0].blocks == [textBlock]

    memory.clear()
    assert len(memory) == 0

    memory.new_turn()
    assert len(memory) == 1

    memory.add_to_last_turn(textBlock)
    assert memory[0].blocks == [textBlock]

    memory.add_to_last_turn(textBlock2)
    assert memory[0].blocks == [textBlock, textBlock2]

    assert len(memory) == 1


def test_client_with_memory():
    text1 = "Mock response"
    text2 = "secondresponse"

    # Creazione Memory
    memory = Memory()

    # Creazione client
    llm = MockClient(model_name="mock_client")
    response = llm.invoke(input=text1, memory=memory)

    assert response.content == [TextBlock(content=text1)]

    assert response.text == text1

    memory.add_turn(response.content, role=ROLE.ASSISTANT)

    response = llm.invoke(input=text2, memory=memory)
    assert response.text == text1 + text2


def test_response_format():
    class TestModel(BaseModel):
        test: int

    input = '{"test": 1}'

    llm = MockClient(model_name="mock_client")
    response = llm.structured_response(input=input, output_cls=TestModel)

    # assert isinstance(response, TestModel)
    # assert response == TestModel(test=1)

    assert response.structured_data == [TestModel(test=1)]


def test_stream_structured_response():
    class TestModel(BaseModel):
        test: int

    input = '{"test": 1}'

    llm = MockClient(model_name="mock_client")
    chunks = list(
        llm.stream_structured_response(input=input, output_cls=TestModel)
    )

    for chunk in chunks[:-1]:
        assert chunk.delta
        assert chunk.content == []
    assert "".join(chunk.delta for chunk in chunks[:-1]) == input

    final = chunks[-1]
    assert final.delta is None
    assert final.structured_data == [TestModel(test=1)]
    assert final.stop_reason == "stop"
    assert final.usage.completion_tokens > 0


def test_a_stream_structured_response():
    class TestModel(BaseModel):
        test: int

    input = '{"test": 1}'

    llm = MockClient(model_name="mock_client")

    async def collect():
        return [
            chunk
            async for chunk in llm.a_stream_structured_response(
                input=input, output_cls=TestModel
            )
        ]

    chunks = asyncio.run(collect())

    for chunk in chunks[:-1]:
        assert chunk.delta
        assert chunk.content == []
    assert "".join(chunk.delta for chunk in chunks[:-1]) == input

    final = chunks[-1]
    assert final.delta is None
    assert final.structured_data == [TestModel(test=1)]
    assert final.stop_reason == "stop"


def test_stream_structured_response_invalid_output_cls():
    llm = MockClient(model_name="mock_client")

    # Errors surface at first iteration because the method is a generator.
    with pytest.raises(ValueError, match="output_cls must be a subclass"):
        list(llm.stream_structured_response(input="{}", output_cls=dict))


def test_stream_structured_response_not_implemented_default():
    class NoStreamStructuredClient(MockClient):
        def _stream_structured_response(self, *args, **kwargs):
            return Client._stream_structured_response(self, *args, **kwargs)

        def _a_stream_structured_response(self, *args, **kwargs):
            return Client._a_stream_structured_response(self, *args, **kwargs)

    class TestModel(BaseModel):
        test: int

    llm = NoStreamStructuredClient(model_name="mock_client")

    with pytest.raises(NotImplementedError):
        list(llm.stream_structured_response(input="{}", output_cls=TestModel))

    async def collect():
        async for _ in llm.a_stream_structured_response(
            input="{}", output_cls=TestModel
        ):
            pass

    # Must be NotImplementedError, not TypeError: the base async default is
    # an async generator function, so `async for` works and raises properly.
    with pytest.raises(NotImplementedError):
        asyncio.run(collect())


def test_stream_structured_response_invalid_json():
    class TestModel(BaseModel):
        test: int

    llm = MockClient(model_name="mock_client")

    with pytest.raises(ValidationError):
        list(llm.stream_structured_response(input="not json", output_cls=TestModel))


def test_stream_structured_response_snapshot_accumulates():
    class TestModel(BaseModel):
        test: int

    input = '{"test": 1}'

    llm = MockClient(model_name="mock_client")
    chunks = list(llm.stream_structured_response(input=input, output_cls=TestModel))

    previous = ""
    for chunk in chunks[:-1]:
        assert chunk.snapshot.startswith(previous)
        assert len(chunk.snapshot) > len(previous)
        assert input.startswith(chunk.snapshot)
        previous = chunk.snapshot

    assert chunks[-1].snapshot == input
    assert chunks[-1].partial == {"test": 1}


def test_a_stream_structured_response_snapshot_accumulates():
    class TestModel(BaseModel):
        test: int

    input = '{"test": 1}'

    llm = MockClient(model_name="mock_client")

    async def collect():
        return [
            chunk
            async for chunk in llm.a_stream_structured_response(
                input=input, output_cls=TestModel
            )
        ]

    chunks = asyncio.run(collect())

    previous = ""
    for chunk in chunks[:-1]:
        assert chunk.snapshot.startswith(previous)
        assert input.startswith(chunk.snapshot)
        previous = chunk.snapshot

    assert chunks[-1].snapshot == input
    assert chunks[-1].partial == {"test": 1}


def test_partial_parses_incomplete_buffer():
    response = ClientResponse(
        content=[], delta="x", snapshot='{"test": 1, "name": "Al'
    )

    assert response.partial == {"test": 1, "name": "Al"}


def test_partial_none_on_missing_or_malformed_snapshot():
    assert ClientResponse(content=[]).partial is None
    assert ClientResponse(content=[], snapshot="").partial is None
    assert ClientResponse(content=[], snapshot="not json").partial is None


def test_partial_json_object_sentinel_stream():
    class JsonObjectStreamMockClient(MockClient):
        def _stream_structured_response(self, input, output_cls, *args, **kwargs):
            yield ClientResponse(content=[], delta='{"k": ')
            yield ClientResponse(content=[], delta='"v"}')
            yield ClientResponse(
                content=[StructuredBlock(content={"k": "v"})],
                delta=None,
                stop_reason="stop",
            )

    llm = JsonObjectStreamMockClient(model_name="mock_client")
    chunks = list(
        llm.stream_structured_response(
            input="anything", output_cls={"type": "json_object"}
        )
    )

    assert chunks[0].partial == {}
    assert chunks[1].partial == {"k": "v"}
    assert chunks[-1].partial == {"k": "v"}
    assert chunks[-1].structured_data == [{"k": "v"}]


def test_to_dict_excludes_snapshot():
    response = ClientResponse(
        content=[TextBlock(content="hi")], snapshot='{"test": 1}'
    )

    data = response.to_dict()
    assert "snapshot" not in data
    assert ClientResponse.from_dict(data).snapshot is None


def test_response_first_text():
    response = ClientResponse(
        content=[TextBlock(content="string1"), TextBlock("text2")]
    )
    assert response.first_text == "string1"


def test_stream_structured_response_module():
    class TestModel(BaseModel):
        test: int

    input = '{"test": 1}'

    llm = MockClient(model_name="mock_client")
    llm_module = llm.as_stream_structured_response_module_component()
    chunks = list(llm_module._run(input=input, output_cls=TestModel))

    assert chunks[-1].structured_data == [TestModel(test=1)]


def test_client_module():
    llm = MockClient(model_name="mock_client")
    llm_module = llm.as_module_component()
    response = llm_module._run(input="Mock response")
    assert isinstance(response, ClientResponse)
    assert response.content == [TextBlock(content="Mock response")]

    memory = Memory()
    memory.add_turn(TextBlock(content="old_message"), role=ROLE.ASSISTANT)
    response_with_memory = llm_module._run(input="Mock response", memory=memory)
    assert "old_message" in response_with_memory.text
    assert "Mock response" in response_with_memory.text
