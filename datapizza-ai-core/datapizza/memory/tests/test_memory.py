import json
from typing import TypeVar

from pydantic import BaseModel

from datapizza.memory.memory import Memory
from datapizza.tools import tool
from datapizza.type import (
    ROLE,
    FunctionCallBlock,
    FunctionCallResultBlock,
    Media,
    MediaBlock,
    StructuredBlock,
    TextBlock,
)

Model = TypeVar("Model", bound=BaseModel)


def test_memory_json_dumps():
    memory = Memory()
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.ASSISTANT)

    json = memory.json_dumps()
    assert (
        json
        == '[{"role": "user", "blocks": [{"type": "text", "content": "Hello, world!"}]}, {"role": "assistant", "blocks": [{"type": "text", "content": "Hello, world!"}]}]'
    )


def test_memory_to_dict():
    memory = Memory()
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.ASSISTANT)

    assert memory.to_dict() == [
        {"role": "user", "blocks": [{"type": "text", "content": "Hello, world!"}]},
        {"role": "assistant", "blocks": [{"type": "text", "content": "Hello, world!"}]},
    ]


def test_to_dict_with_media_block():
    memory = Memory()
    memory.add_turn(
        blocks=[
            MediaBlock(
                media=Media(
                    media_type="image",
                    source_type="url",
                    source="https://example.com/image.png",
                    extension="png",
                )
            )
        ],
        role=ROLE.USER,
    )

    assert memory.to_dict() == [
        {
            "role": "user",
            "blocks": [
                {
                    "type": "media",
                    "media": {
                        "source": "https://example.com/image.png",
                        "extension": "png",
                        "media_type": "image",
                        "source_type": "url",
                        "detail": "high",
                    },
                }
            ],
        },
    ]


def test_to_dict_with_media_block_base64():
    memory = Memory()
    memory.add_turn(
        blocks=[
            MediaBlock(
                media=Media(
                    media_type="image",
                    source_type="base64",
                    source="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQAQMAAAAlPW0iAAAABlBMVEX///+/v7+jQ3Y5AAAACklEQVQI12NgAAAAAgAB4iG8MwAAAABJRU5ErkJggg==",
                    extension="png",
                )
            )
        ],
        role=ROLE.USER,
    )

    assert memory.to_dict() == [
        {
            "role": "user",
            "blocks": [
                {
                    "type": "media",
                    "media": {
                        "source": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQAQMAAAAlPW0iAAAABlBMVEX///+/v7+jQ3Y5AAAACklEQVQI12NgAAAAAgAB4iG8MwAAAABJRU5ErkJggg==",
                        "extension": "png",
                        "media_type": "image",
                        "source_type": "base64",
                        "detail": "high",
                    },
                }
            ],
        },
    ]


def test_memory_json_dumps_with_function_call():
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city"""
        return f"The weather in {city} is sunny"

    memory = Memory()
    memory.add_turn(
        blocks=[
            FunctionCallBlock(
                id="1",
                arguments="{}",
                name="get_weather",
                tool=get_weather,
            )
        ],
        role=ROLE.USER,
    )

    assert memory.to_dict() == [
        {
            "role": "user",
            "blocks": [
                {
                    "type": "function",
                    "id": "1",
                    "arguments": "{}",
                    "name": "get_weather",
                    "tool": {
                        "name": "get_weather",
                        "description": "Get the weather for a city",
                        "properties": {
                            "city": {"type": "string", "description": "Parameter city"}
                        },
                        "required": ["city"],
                        "end_invoke": False,
                    },
                }
            ],
        }
    ]


def test_memory_json_dumps_with_function_call_result():
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city"""
        return f"The weather in {city} is sunny"

    memory = Memory()

    memory.add_turn(
        blocks=[
            FunctionCallResultBlock(
                id="1",
                result="The weather in Rome is sunny",
                tool=get_weather,
            )
        ],
        role=ROLE.ASSISTANT,
    )

    assert memory.to_dict() == [
        {
            "role": "assistant",
            "blocks": [
                {
                    "type": "function_call_result",
                    "id": "1",
                    "tool": {
                        "name": "get_weather",
                        "description": "Get the weather for a city",
                        "properties": {
                            "city": {"type": "string", "description": "Parameter city"}
                        },
                        "required": ["city"],
                        "end_invoke": False,
                    },
                    "result": "The weather in Rome is sunny",
                }
            ],
        }
    ]


def test_memory_json_dumps_with_function_call_bytes_thought_signature():
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city"""
        return f"The weather in {city} is sunny"

    memory = Memory()
    memory.add_turn(
        blocks=[
            FunctionCallBlock(
                id="1",
                arguments={"city": "Rome"},
                name="get_weather",
                tool=get_weather,
                thought_signature=b"\x01\x02\x03",
            )
        ],
        role=ROLE.ASSISTANT,
    )

    dumped = memory.json_dumps()
    payload = json.loads(dumped)

    assert payload[0]["blocks"][0]["thought_signature"] == {
        "encoding": "base64",
        "data": "AQID",
    }


def test_structured_block():
    class Person(BaseModel):
        name: str
        age: int

    memory = Memory()
    memory.add_turn(
        blocks=[StructuredBlock(content=Person(name="John", age=30))],
        role=ROLE.ASSISTANT,
    )

    mem_dict = memory.to_dict()
    assert mem_dict == [
        {
            "role": "assistant",
            "blocks": [
                {
                    "type": "structured",
                    "content": '{"name":"John","age":30}',
                }
            ],
        }
    ]


def test_structured_json_block():
    memory = Memory()
    memory.add_turn(
        blocks=[StructuredBlock(content={"name": "John", "age": 30})],
        role=ROLE.ASSISTANT,
    )

    mem_dict = memory.to_dict()
    assert mem_dict == [
        {
            "role": "assistant",
            "blocks": [
                {
                    "type": "structured",
                    "content": {"name": "John", "age": 30},
                }
            ],
        }
    ]


def test_memory_copy():
    memory = Memory()
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.ASSISTANT)

    memory_copy = memory.copy()
    assert memory_copy == memory

    memory_copy.clear()
    assert memory_copy == Memory()


def test_memory_deep_copy():
    memory = Memory()
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.ASSISTANT)

    memory_copy = memory.copy()
    assert memory_copy == memory

    memory[0].blocks[0].content = "Hello, world! 2"
    assert memory_copy[0].blocks[0].content == "Hello, world!"

    memory[0].role = ROLE.ASSISTANT
    assert memory_copy[0].role == ROLE.USER


def test_empty_memory_json_loads():
    memory = Memory()

    json_str = memory.json_dumps()
    memory_copy = Memory()
    memory_copy.json_loads(json_str)
    assert memory_copy == memory


def test_memory_json_loadsas():
    memory = Memory()
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)

    json_str = memory.json_dumps()

    memory_copy = Memory()
    memory_copy.json_loads(json_str)
    assert memory_copy == memory
