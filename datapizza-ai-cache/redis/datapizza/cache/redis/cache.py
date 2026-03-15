import logging

from datapizza.core.cache import Cache

import redis

from .json_codec import decode_cache_value, encode_cache_value

log = logging.getLogger(__name__)


class RedisCache(Cache):
    """
    A Redis-based cache implementation.
    """

    def __init__(
        self, host="localhost", port=6379, db=0, expiration_time=3600
    ):  # 1 hour default
        self.redis = redis.Redis(host=host, port=port, db=db)
        self.expiration_time = expiration_time

    def get(self, key: str) -> object | None:
        """Retrieve and deserialize object"""
        raw_value = self.redis.get(key)
        if raw_value is None:
            return None

        try:
            return decode_cache_value(raw_value)
        except Exception as exc:
            log.warning("Invalid cache payload for key %s: %s", key, exc)
            return None

    def set(self, key: str, obj: object):
        """Serialize and store object"""
        serialized = encode_cache_value(obj)
        self.redis.set(key, serialized, ex=self.expiration_time)
