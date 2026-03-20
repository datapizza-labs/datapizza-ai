import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datapizza.type import StructuredBlock, TextBlock
from pydantic import BaseModel

from datapizza.clients.anthropic import AnthropicClient


class _Person(BaseModel):
    name: str
    age: int


@pytest.fixture
def mock_usage():
    u = MagicMock()
    u.input_tokens = 12
    u.output_tokens = 34
    u.cache_read_input_tokens = 0
    return u


@patch("datapizza.clients.anthropic.anthropic_client.Anthropic")
def test_structured_response_parse_sync(mock_anthropic_class, mock_usage):
    mock_instance = mock_anthropic_class.return_value
    parsed_msg = MagicMock()
    parsed_msg.parsed_output = _Person(name="Ada", age=36)
    parsed_msg.stop_reason = "end_turn"
    parsed_msg.usage = mock_usage
    mock_instance.messages.parse.return_value = parsed_msg

    client = AnthropicClient(api_key="key", model="claude-sonnet-4-20250514")
    result = client._structured_response(
        input=[TextBlock(content="Who?")],
        output_cls=_Person,
        max_tokens=512,
    )

    mock_instance.messages.parse.assert_called_once()
    call_kw = mock_instance.messages.parse.call_args.kwargs
    assert call_kw["model"] == "claude-sonnet-4-20250514"
    assert call_kw["output_format"] is _Person
    assert call_kw["max_tokens"] == 512

    assert len(result.content) == 1
    assert isinstance(result.content[0], StructuredBlock)
    assert result.content[0].content == _Person(name="Ada", age=36)
    assert result.prompt_tokens_used == 12
    assert result.completion_tokens_used == 34


@patch("datapizza.clients.anthropic.anthropic_client.AsyncAnthropic")
def test_structured_response_parse_async(mock_async_class, mock_usage):
    mock_instance = mock_async_class.return_value
    parsed_msg = MagicMock()
    parsed_msg.parsed_output = _Person(name="Bob", age=40)
    parsed_msg.stop_reason = "end_turn"
    parsed_msg.usage = mock_usage
    mock_instance.messages.parse = AsyncMock(return_value=parsed_msg)

    client = AnthropicClient(api_key="key", model="claude-sonnet-4-20250514")

    async def run():
        return await client._a_structured_response(
            input=[TextBlock(content="Hi")],
            output_cls=_Person,
        )

    result = asyncio.run(run())

    mock_instance.messages.parse.assert_awaited_once()
    assert result.content[0].content.name == "Bob"
    assert result.stop_reason == "end_turn"


@patch("datapizza.clients.anthropic.anthropic_client.Anthropic")
def test_structured_response_json_object_sentinel(mock_anthropic_class, mock_usage):
    mock_instance = mock_anthropic_class.return_value
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = '{"foo": 42}'
    raw_msg = MagicMock()
    raw_msg.content = [text_block]
    raw_msg.stop_reason = "end_turn"
    raw_msg.usage = mock_usage
    mock_instance.messages.create.return_value = raw_msg

    client = AnthropicClient(api_key="key", model="claude-haiku")
    result = client._structured_response(
        input=[TextBlock(content="return json")],
        output_cls={"type": "json_object"},
    )

    mock_instance.messages.create.assert_called_once()
    assert mock_instance.messages.parse.call_count == 0
    kw = mock_instance.messages.create.call_args.kwargs
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert result.content[0].content == {"foo": 42}
