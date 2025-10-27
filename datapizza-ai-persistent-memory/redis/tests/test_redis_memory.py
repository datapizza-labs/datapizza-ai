import pytest
from datapizza.persistent_memory.redis import RedisMemory

from datapizza.type import ROLE, TextBlock


@pytest.fixture
def redis_memory():
    return RedisMemory(user_id="test_user", session_id="test_session")


def test_redis_memory_initialization(redis_memory):
    assert isinstance(redis_memory, RedisMemory)
    assert redis_memory.user_id == "test_user"
    assert redis_memory.session_id == "test_session"


def test_redis_memory_new_turn(redis_memory):
    redis_memory.new_turn(role=ROLE.USER)
    assert len(redis_memory) == 1
    assert redis_memory[0].role == ROLE.USER


def test_redis_memory_add_turn(redis_memory):
    redis_memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    assert len(redis_memory) == 1
    assert redis_memory[0].blocks[0].content == "Hello, world!"


def test_redis_memory_add_to_last_turn(redis_memory):
    redis_memory.new_turn(role=ROLE.USER)
    redis_memory.add_to_last_turn(TextBlock(content="Hello, world!"))
    assert len(redis_memory) == 1
    assert redis_memory[0].blocks[0].content == "Hello, world!"


def test_redis_memory_clear(redis_memory):
    redis_memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    redis_memory.clear()
    assert len(redis_memory) == 0


def test_redis_memory_iter(redis_memory):
    redis_memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    redis_memory.add_turn(
        blocks=[TextBlock(content="How are you?")], role=ROLE.ASSISTANT
    )
    turns = list(redis_memory)
    assert len(turns) == 2
    assert turns[0].blocks[0].content == "Hello, world!"
    assert turns[1].blocks[0].content == "How are you?"


def test_redis_memory_iter_blocks(redis_memory):
    redis_memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    redis_memory.add_turn(
        blocks=[TextBlock(content="How are you?")], role=ROLE.ASSISTANT
    )
    blocks = list(redis_memory.iter_blocks())
    assert len(blocks) == 2
    assert blocks[0].content == "Hello, world!"
    assert blocks[1].content == "How are you?"


def test_redis_memory_getitem(redis_memory):
    redis_memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    turn = redis_memory[0]
    assert turn.blocks[0].content == "Hello, world!"


def test_redis_memory_setitem(redis_memory):
    redis_memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    redis_memory[0] = [TextBlock(content="Updated content")]
    assert redis_memory[0].blocks[0].content == "Updated content"


def test_redis_memory_delitem(redis_memory):
    redis_memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    del redis_memory[0]
    assert len(redis_memory) == 0


def test_redis_memory_json_dumps(redis_memory):
    redis_memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    json_str = redis_memory.json_dumps()
    assert (
        json_str
        == '[{"role": "user", "blocks": [{"type": "text", "content": "Hello, world!"}]}]'
    )


def test_redis_memory_json_loads(redis_memory):
    json_str = (
        '[{"role": "user", "blocks": [{"type": "text", "content": "Hello, world!"}]}]'
    )
    redis_memory.json_loads(json_str)
    assert len(redis_memory) == 1
    assert redis_memory[0].blocks[0].content == "Hello, world!"


def test_redis_memory_to_dict(redis_memory):
    redis_memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    assert redis_memory.to_dict() == [
        {"role": "user", "blocks": [{"type": "text", "content": "Hello, world!"}]}
    ]


def test_redis_memory_copy(redis_memory):
    redis_memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    redis_memory_copy = redis_memory.copy()
    assert redis_memory_copy == redis_memory


def test_redis_memory_deep_copy(redis_memory):
    redis_memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    redis_memory_copy = redis_memory.copy()
    redis_memory[0].blocks[0].content = "Updated content"
    assert redis_memory_copy[0].blocks[0].content == "Hello, world!"


def test_redis_memory_equality(redis_memory):
    redis_memory1 = RedisMemory(user_id="test_user", session_id="test_session")
    redis_memory2 = RedisMemory(user_id="test_user", session_id="test_session")
    assert redis_memory1 == redis_memory2
    redis_memory1.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)

    assert len(redis_memory2) == 1


def test_redis_memory_inequality(redis_memory):
    redis_memory1 = RedisMemory(user_id="test_user", session_id="test_session")
    redis_memory2 = RedisMemory(user_id="test_user", session_id="different_session")
    assert redis_memory1 != redis_memory2
    redis_memory1.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)

    assert len(redis_memory2) == 1
