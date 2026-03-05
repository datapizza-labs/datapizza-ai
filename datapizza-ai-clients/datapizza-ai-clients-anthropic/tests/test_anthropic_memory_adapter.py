from datapizza.memory.memory import Memory
from datapizza.tools.tools import tool
from datapizza.type import ROLE, FunctionCallBlock, FunctionCallResultBlock, TextBlock

from datapizza.clients.anthropic import AnthropicClient
from datapizza.clients.anthropic.memory_adapter import AnthropicMemoryAdapter


def test_init_anthropic_client():
    client = AnthropicClient(api_key="test")
    assert client is not None


def test_anthropic_memory_adapter():
    memory_adapter = AnthropicMemoryAdapter()
    memory = Memory()
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.ASSISTANT)

    messages = memory_adapter.memory_to_messages(memory)

    assert messages == [
        {
            "role": "user",
            "content": "Hello, world!",
        },
        {
            "role": "assistant",
            "content": "Hello, world!",
        },
    ]


def test_anthropic_memory_adapter_tool_result_turn_maps_to_user_role():
    @tool
    def get_city() -> str:
        return "Rome"

    memory_adapter = AnthropicMemoryAdapter()
    memory = Memory()
    memory.add_turn(
        blocks=[FunctionCallResultBlock(id="toolu_1", tool=get_city, result="Rome")],
        role=ROLE.TOOL,
    )

    messages = memory_adapter.memory_to_messages(memory)

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "Rome",
        }
    ]


def test_anthropic_memory_adapter_function_call_turn_maps_to_assistant_tool_use():
    @tool
    def get_city() -> str:
        return "Rome"

    memory_adapter = AnthropicMemoryAdapter()
    memory = Memory()
    memory.add_turn(
        blocks=[
            FunctionCallBlock(
                id="toolu_1",
                name="get_city",
                arguments={"country": "Italy"},
                tool=get_city,
            )
        ],
        role=ROLE.ASSISTANT,
    )

    messages = memory_adapter.memory_to_messages(memory)

    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_city",
            "input": {"country": "Italy"},
        }
    ]
