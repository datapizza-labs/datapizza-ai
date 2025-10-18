import time
from datapizza.cache.redis import RedisCache


def test_redis_cache_set():
    """
    Unit test for RedisCache.set() method.
    Verifies that data is correctly serialized and stored in Redis with TTL.
    """
    cache = RedisCache(host="localhost", port=6379, db=0, expiration_time=3600)
    
    # Arrange
    test_key = "test_set_key"
    test_value = {"name": "Alice", "age": 30, "tags": ["python", "ai"]}
    
    # Act
    cache.set(test_key, test_value)
    
    # Assert
    assert cache.redis.exists(test_key) == 1, "Key should exist in Redis after set()"
    ttl = cache.redis.ttl(test_key)
    assert ttl > 0, f"TTL should be positive after set(), got {ttl}"
    
    # Cleanup
    cache.redis.delete(test_key)


def test_redis_cache_get():
    """
    Unit test for RedisCache.get() method.
    Verifies that data is correctly retrieved and deserialized from Redis.
    """
    cache = RedisCache(host="localhost", port=6379, db=0)
    
    # Arrange
    test_key = "test_get_key"
    test_value = [1, 2, 3, 4, 5]
    cache.set(test_key, test_value)
    
    # Act
    retrieved_value = cache.get(test_key)
    
    # Assert
    assert retrieved_value == test_value, "Retrieved value should match stored value"
    assert isinstance(retrieved_value, list), "Retrieved value should be a list"
    
    # Cleanup
    cache.redis.delete(test_key)
