import json
from typing import cast

from datapizza.core.clients.models import ClientResponse, TokenUsage
from datapizza.tools import Tool
from datapizza.type import FunctionCallBlock, TextBlock

from datapizza.cache.redis import RedisCache


class _FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value


def _build_cache(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(
        "datapizza.cache.redis.cache.redis.Redis", lambda **kwargs: fake
    )
    cache = RedisCache(host="localhost", port=6379, db=0)
    return cache, fake


def test_redis_cache_initializes(monkeypatch):
    cache, _ = _build_cache(monkeypatch)
    assert cache is not None


def test_roundtrip_client_response(monkeypatch):
    cache, _ = _build_cache(monkeypatch)
    tool = Tool(name="calc", description="calculator")
    response = ClientResponse(
        content=[
            TextBlock(content="hello"),
            FunctionCallBlock(id="f1", arguments={"x": 1}, name="calc", tool=tool),
        ],
        delta="h",
        stop_reason="end",
        usage=TokenUsage(
            prompt_tokens=1, completion_tokens=2, cached_tokens=3, thinking_tokens=4
        ),
    )

    cache.set("k1", response)
    cached = cache.get("k1")

    assert isinstance(cached, ClientResponse)
    cached_response = cast(ClientResponse, cached)
    assert cached_response.text == "hello"
    assert len(cached_response.function_calls) == 1
    assert cached_response.usage == response.usage
    assert cached_response.stop_reason == "end"


def test_roundtrip_json_value(monkeypatch):
    cache, _ = _build_cache(monkeypatch)
    value = [[0.1, 0.2], [0.3, 0.4]]

    cache.set("embed", value)
    cached = cache.get("embed")

    assert cached == value


def test_invalid_payload_is_cache_miss(monkeypatch):
    cache, fake = _build_cache(monkeypatch)
    fake.store["broken"] = b"not-json"

    assert cache.get("broken") is None


def test_unknown_payload_version_is_cache_miss(monkeypatch):
    cache, fake = _build_cache(monkeypatch)
    fake.store["old"] = json.dumps({"v": 999, "t": "json", "p": 1}).encode("utf-8")

    assert cache.get("old") is None


def test_legacy_pickle_like_bytes_are_ignored(monkeypatch):
    cache, fake = _build_cache(monkeypatch)
    fake.store["legacy"] = b"\x80\x04legacy-pickle-bytes"

    assert cache.get("legacy") is None
