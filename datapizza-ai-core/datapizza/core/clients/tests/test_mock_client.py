import pytest
from pydantic import BaseModel

from datapizza.clients.mock_client import MockClient
from datapizza.core.clients import Client, ClientResponse
from datapizza.memory.memory import Memory
from datapizza.type import TextBlock
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


def test_response_first_text():
    response = ClientResponse(
        content=[TextBlock(content="string1"), TextBlock("text2")]
    )
    assert response.first_text == "string1"


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
