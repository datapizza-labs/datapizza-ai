import datetime
import json

import redis
from redis.commands.json.path import Path
from redis.commands.search.field import TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from datapizza.memory import Memory, Turn
from datapizza.type import Block, FunctionCallBlock, ROLE


class RedisMemory(Memory):
    """
    A Redis-based memory implementation that extends the original Memory class.
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        host="localhost",
        port=6379,
        db=0,
        expiration_time=3600,
        force_new_index: bool = False,
        index_name: str = "idx:history",
    ):
        self.redis = redis.Redis(host=host, port=port, db=db,decode_responses=True)

        self.expiration_time = expiration_time
        self.user_id = user_id
        self.session_id = session_id
        self.key_prefix = f"history:{self.user_id}:{self.session_id}"
        self.index_name = index_name

        # Create Redis index for chat session messages
        if force_new_index:
            self.redis.ft(index_name).dropindex(True)
        try:
            schema = (
                TextField("$.user_id", as_name="user_id"),
                TextField("$.session_id", as_name="session_id"),
                TextField("$.content", as_name="content"),
                TextField("$.ts", as_name="timestamp", sortable=True),

            )
            indexCreated = self.redis.ft(index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=["history:"], index_type=IndexType.JSON
                )
            )
        except redis.exceptions.ResponseError:
            pass

    @property
    def _next_index(self):
        return f"{self.key_prefix}:next_index"

    def _get_turn_key(self, index: int) -> str:
        return f"{self.key_prefix}:{index}"

    def _get_next_index(self) -> int:

        if not self.redis.exists(self._next_index):
            self.redis.set(self._next_index, 0)
        return self.redis.incr(self._next_index)

    def _get_last_index(self):

        if not self.redis.exists(self._next_index):
            self.redis.set(self._next_index, 0)
        return int(self.redis.get(self._next_index))

    def _serialize_turn(self, turn: Turn) -> str:
        return json.dumps(turn.to_dict())

    def _deserialize_turn(self, turn_str: str) -> Turn:
        turn_dict = json.loads(turn_str)
        blocks = [Block.from_dict(block) for block in turn_dict["blocks"]]
        return Turn(blocks, ROLE(turn_dict["role"]))

    def new_turn(self, role: ROLE = ROLE.ASSISTANT):
        index = self._get_next_index()
        turn = Turn([], role)
        self.redis.set(
            self._get_turn_key(index),
            self._serialize_turn(turn),
            ex=self.expiration_time,
        )

    def add_turn(
        self, blocks: list[Block] | list[FunctionCallBlock] | Block, role: ROLE
    ):
        turn = Turn(blocks, role) if isinstance(blocks, list) else Turn([blocks], role)  # type: ignore
        index = self._get_next_index()
        turn_key = self._get_turn_key(index)

        # Add turn to Redis index
        user1Set = self.redis.json().set(f"{turn_key}", Path.root_path(), {"session_id": self.session_id,
                                                                           "user_id": self.user_id,
                                                                           "content": self._serialize_turn(turn),
                                                                           "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})




    def add_to_last_turn(self, block: Block):
        if not self._get_next_index():
            self.new_turn()
        index = self._get_next_index() - 1
        turn = self._deserialize_turn(self.redis.get(self._get_turn_key(index)))
        turn.append(block)
        self.redis.set(
            self._get_turn_key(index),
            self._serialize_turn(turn),
            ex=self.expiration_time,
        )

    def clear(self):
        for index in range(self._get_last_index() + 1) :
            self.redis.delete(self._get_turn_key(index))
        self.redis.delete(self._next_index)


    def copy(self):
        """Deep copy the memory."""
        new_memory = RedisMemory(user_id=self.user_id, session_id=self.session_id, host=self.redis.connection_pool.connection_kwargs['host'], port=self.redis.connection_pool.connection_kwargs['port'], db=self.redis.connection_pool.connection_kwargs['db'], expiration_time=self.expiration_time)

        return new_memory
    def __iter__(self):
        for index in range(self._get_last_index()):
            turn_key = self._get_turn_key(index)
            turn_str = self.redis.json().get(turn_key)
            if turn_str:
                yield self._deserialize_turn(turn_str["content"])

    def iter_blocks(self):
        for turn in self:
            yield from turn

    def __len__(self):
        return self._get_last_index()

    def __getitem__(self, index):
        turn_key = self._get_turn_key(index)
        turn_str = self.redis.json().get(turn_key)
        if turn_str:
            return self._deserialize_turn(turn_str["content"])
        return None

    def __setitem__(self, index, value):
        if isinstance(value, list):
            turn = Turn(value, ROLE.ASSISTANT)
        else:
            turn = Turn([value], ROLE.ASSISTANT)
        self.redis.set(
            self._get_turn_key(index),
            self._serialize_turn(turn),
            ex=self.expiration_time,
        )

    def __delitem__(self, index):
        self.redis.delete(self._get_turn_key(index))

    def __str__(self):
        return str(list(self.memory))

    def __repr__(self):
        return f"RedisMemory(turns={len(self)})"

    def __bool__(self):
        return bool(self.redis)

    def __eq__(self, other):
        if not isinstance(other, RedisMemory):
            return False
        return hash(self) == hash(other)

    def __hash__(self):
        return super().__hash__()

    @property
    def memory(self) -> list[Turn]:
        query = f"@user_id:{self.user_id} @session_id:{self.session_id}"
        results =  self.redis.ft(self.index_name).search(
            Query(query).sort_by("timestamp", asc=True)
        )
        turns = []
        for result in results.docs:
            turns.append(json.loads(result['json'])["content"])
        return turns

    def json_dumps(self) -> str:
        return json.dumps([turn.to_dict() for turn in self.memory])

    def json_loads(self, json_str: str):
        obj = json.loads(json_str)
        for t in obj:
            self.add_turn(
                blocks=[Block.from_dict(block) for block in t["blocks"]],
                role=ROLE(t["role"]),
            )

    def to_dict(self) -> list[dict]:
        return [turn.to_dict() for turn in self.memory]
