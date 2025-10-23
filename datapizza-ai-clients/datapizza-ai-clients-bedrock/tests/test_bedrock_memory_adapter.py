from datapizza.memory.memory import Memory
from datapizza.type import ROLE, TextBlock

from datapizza.clients.bedrock import BedrockClient
from datapizza.clients.bedrock.memory_adapter import BedrockMemoryAdapter


def test_init_bedrock_client():
    """Test that BedrockClient can be initialized"""
    client = BedrockClient()
    assert client is not None
    assert client.model_name == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_init_bedrock_client_with_credentials():
    """Test BedrockClient initialization with explicit credentials"""
    client = BedrockClient(
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
        region_name="us-west-2",
    )
    assert client is not None
    assert client.aws_access_key_id == "test_key"
    assert client.region_name == "us-west-2"


def test_bedrock_memory_adapter():
    """Test that the memory adapter converts memory to Bedrock message format"""
    memory_adapter = BedrockMemoryAdapter()
    memory = Memory()
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    memory.add_turn(
        blocks=[TextBlock(content="Hello! How can I help you?")], role=ROLE.ASSISTANT
    )

    messages = memory_adapter.memory_to_messages(memory)

    assert messages == [
        {
            "role": "user",
            "content": [{"text": "Hello, world!"}],
        },
        {
            "role": "assistant",
            "content": [{"text": "Hello! How can I help you?"}],
        },
    ]


def test_bedrock_memory_adapter_multiple_blocks():
    """Test memory adapter with multiple text blocks in a single turn"""
    memory_adapter = BedrockMemoryAdapter()
    memory = Memory()
    memory.add_turn(
        blocks=[
            TextBlock(content="First message."),
            TextBlock(content="Second message."),
        ],
        role=ROLE.USER,
    )

    messages = memory_adapter.memory_to_messages(memory)

    assert messages == [
        {
            "role": "user",
            "content": [{"text": "First message."}, {"text": "Second message."}],
        },
    ]
