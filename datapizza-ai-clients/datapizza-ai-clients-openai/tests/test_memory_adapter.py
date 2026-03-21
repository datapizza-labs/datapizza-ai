import json

import pytest
from datapizza.memory.memory import Memory
from datapizza.tools.tools import tool
from datapizza.type.type import (
    ROLE,
    FunctionCallBlock,
    Media,
    MediaBlock,
    StructuredBlock,
    TextBlock,
)
from openai.types.responses import ResponseFunctionToolCall

from datapizza.clients.openai.memory_adapter import OpenAIMemoryAdapter


@pytest.fixture(
    params=[
        OpenAIMemoryAdapter(),
    ]
)
def adapter(request):
    """Parameterized fixture that provides different memory adapter implementations.

    Each test using this fixture will run once for each adapter in the params list.
    """
    return request.param


@pytest.fixture
def memory():
    return Memory()


def test_empty_memory_to_messages(adapter, memory):
    """Test that an empty memory converts to an empty list of messages."""
    messages = adapter.memory_to_messages(memory)
    assert messages == []


def test_turn_with_some_text():
    memory = Memory()
    memory.new_turn(role=ROLE.USER)
    memory.add_to_last_turn(TextBlock(content="Hello!"))
    memory.add_to_last_turn(TextBlock(content="Hi, how are u?"))
    messages = OpenAIMemoryAdapter().memory_to_messages(memory)
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Hello!"},
                {"type": "input_text", "text": "Hi, how are u?"},
            ],
        }
    ]


def test_memory_to_messages_multiple_turns():
    """Test conversion of a memory with multiple turns to messages."""
    # First turn: user asks a question
    memory = Memory()
    memory.new_turn(role=ROLE.USER)
    memory.add_to_last_turn(TextBlock(content="What's 2+2?"))

    # Second turn: assistant responds
    memory.new_turn(role=ROLE.ASSISTANT)
    memory.add_to_last_turn(TextBlock(content="The answer is 4."))

    # Third turn: user follows up
    memory.new_turn(role=ROLE.USER)
    memory.add_to_last_turn(TextBlock(content="Thanks!"))

    messages = OpenAIMemoryAdapter().memory_to_messages(memory)

    expected = [
        {"role": "user", "content": [{"type": "input_text", "text": "What's 2+2?"}]},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "The answer is 4."}],
        },
        {"role": "user", "content": [{"type": "input_text", "text": "Thanks!"}]},
    ]
    assert messages == expected


def test_memory_to_messages_function_call():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    memory = Memory()
    memory.new_turn(role=ROLE.USER)
    memory.add_to_last_turn(TextBlock(content="Call the add function."))
    memory.new_turn(role=ROLE.ASSISTANT)
    memory.add_to_last_turn(
        FunctionCallBlock(id="call_1", name="add", arguments={"a": 2, "b": 2}, tool=add)
    )

    messages = OpenAIMemoryAdapter().memory_to_messages(memory)
    assert messages[0]["role"] == "user"
    assert isinstance(messages[1], ResponseFunctionToolCall)
    assert json.loads(messages[1].arguments) == {
        "a": 2,
        "b": 2,
    }


def test_memory_to_messages_media_blocks():
    image = Media(
        media_type="image",
        source_type="url",
        source="http://example.com/image.png",
        extension="png",
    )
    pdf = Media(
        media_type="pdf",
        source_type="base64",
        source="THIS_IS_A_PDF_BASE64",
        extension="pdf",
    )
    memory = Memory()
    memory.new_turn(role=ROLE.USER)
    memory.add_to_last_turn(MediaBlock(media=image))
    memory.add_to_last_turn(MediaBlock(media=pdf))
    messages = OpenAIMemoryAdapter().memory_to_messages(memory)
    assert messages[0]["role"] == "user"
    # Should contain both image and pdf blocks

    # TODO: Check if the image and pdf blocks are correct
    assert messages[0]["content"][1] == {
        "type": "input_file",
        "filename": "file.pdf",
        "file_data": "data:application/pdf;base64,THIS_IS_A_PDF_BASE64",
    }


def test_memory_to_messages_structured_block():
    memory = Memory()
    memory.new_turn(role=ROLE.USER)
    memory.add_to_last_turn(StructuredBlock(content={"key": "value"}))
    messages = OpenAIMemoryAdapter().memory_to_messages(memory)
    assert messages[0]["content"] == "{'key': 'value'}" or messages[0]["content"] == [
        {
            "type": "input_text",
            "text": "{'key': 'value'}",
        }
    ]


def test_memory_to_messages_with_system_prompt():
    memory = Memory()
    memory.new_turn(role=ROLE.USER)
    memory.add_to_last_turn(TextBlock(content="Hello!"))
    system_prompt = "You are a helpful assistant."
    messages = OpenAIMemoryAdapter().memory_to_messages(
        memory, system_prompt=system_prompt
    )
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_prompt
    assert messages[1]["role"] == "user"


def test_memory_to_messages_with_input_str():
    memory = Memory()
    input_str = "What is the weather?"
    messages = OpenAIMemoryAdapter().memory_to_messages(memory, input=input_str)
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == input_str


def test_memory_to_messages_with_input_block():
    memory = Memory()
    input_block = TextBlock(content="This is a block input.")
    messages = OpenAIMemoryAdapter().memory_to_messages(memory, input=input_block)
    assert messages[-1]["role"] == "user"
    assert "block input" in str(messages[-1]["content"])


def test_google_empty_memory_to_messages():
    messages = OpenAIMemoryAdapter().memory_to_messages(Memory())
    assert messages == []


def test_google_memory_to_messages_multiple_turns():
    memory = Memory()
    memory.new_turn(role=ROLE.USER)
    memory.add_to_last_turn(TextBlock(content="What's 2+2?"))
    memory.new_turn(role=ROLE.ASSISTANT)
    memory.add_to_last_turn(TextBlock(content="The answer is 4."))
    memory.new_turn(role=ROLE.USER)
    memory.add_to_last_turn(TextBlock(content="Thanks!"))
    messages = OpenAIMemoryAdapter().memory_to_messages(memory)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
