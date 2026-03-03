from datapizza.tools.tools import Tool, tool
from datapizza.type.type import (
    Block,
    FunctionCallBlock,
    FunctionCallResultBlock,
    Media,
    MediaBlock,
    TextBlock,
)


def test_hash():
    block = TextBlock(content="Hello, world!")
    print(hash(block))
    assert hash(block) == hash(block)


def test_hash_function_call_block():
    def test_tool():
        pass

    block = FunctionCallBlock(
        id="1",
        arguments="{}",
        name="test",
        tool=Tool(name="test", func=test_tool),
    )
    print(hash(block))
    assert hash(block) == hash(block)
    assert hash(block) == 418964443453056350


def test_block_new():
    block = Block.from_dict({"type": "text", "content": "Hello, world!"})
    assert isinstance(block, TextBlock)
    assert block.content == "Hello, world!"


def test_block_new_function_call():
    @tool
    def test_func():
        pass

    block = FunctionCallBlock(id="1", arguments={}, name="test", tool=test_func)

    assert isinstance(block, FunctionCallBlock)
    assert block.id == "1"
    assert block.arguments == {}
    assert block.name == "test"

    json_data = block.to_dict()
    Block.from_dict(json_data)

    assert isinstance(block, FunctionCallBlock)
    assert block.id == "1"
    assert isinstance(block.tool, Tool)


def test_block_new_function_call_with_bytes_thought_signature():
    @tool
    def test_func():
        pass

    block = FunctionCallBlock(
        id="1",
        arguments={},
        name="test",
        tool=test_func,
        thought_signature=b"\x01\x02\x03",
    )

    json_data = block.to_dict()
    restored = Block.from_dict(json_data)

    assert json_data["thought_signature"] == {"encoding": "base64", "data": "AQID"}
    assert isinstance(restored, FunctionCallBlock)
    assert restored.thought_signature == b"\x01\x02\x03"


def test_block_new_function_call_result():
    @tool
    def test_func():
        pass

    block = FunctionCallResultBlock(id="1", tool=test_func, result={})

    json_data = block.to_dict()
    Block.from_dict(json_data)

    assert isinstance(block, FunctionCallResultBlock)
    assert block.id == "1"
    assert block.result == {}


def test_block_new_media():
    block = Block.from_dict(
        {
            "type": "media",
            "media": {
                "extension": "png",
                "media_type": "image",
                "source_type": "url",
                "source": "https://example.com/image.png",
                "detail": "high",
            },
        }
    )
    assert isinstance(block, MediaBlock)
    assert isinstance(block.media, Media)


# def test_byte_block():
#     block = ByteBlock(data=b"Hello, world!")
#     assert isinstance(block, ByteBlock)
#     assert block.data == b"Hello, world!"
#
#
# def test_byte_block_hash():
#     block = ByteBlock(data=b"Hello, world!")
#     block2 = ByteBlock(data=b"Hello, world!")
#     assert hash(block) == hash(block2)
