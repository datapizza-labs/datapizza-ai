# DataPizza AI - Redis Cache Implementation

A high-performance Redis-based cache implementation for the DataPizza AI framework. This module provides a distributed, persistent cache solution for storing and retrieving LLM responses and other cached data in your AI applications.

## Overview

**datapizza-ai-cache-redis** is a specialized cache adapter that integrates Redis with the DataPizza AI framework. It extends the base `Cache` abstract class to provide distributed caching capabilities with automatic serialization and expiration management.

## Installation

### Prerequisites

- Python 3.10+
- Redis server running (local or remote)

### Install from PyPI

```bash
pip install datapizza-ai-cache-redis
```

### Install from source

```bash
# Clone the repository
git clone https://github.com/datapizza-labs/datapizza-ai.git
cd datapizza-ai/datapizza-ai-cache/redis

# Install in development mode
pip install -e .
```

### Start Redis Locally (Optional)

For local development, you can run Redis using Docker:

```bash
docker run -d -p 6379:6379 redis:latest
```

Or install Redis natively:

**macOS**:
```bash
brew install redis
redis-server
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get install redis-server
redis-server
```

## Quick Start

### Basic Usage

```python
from datapizza.cache.redis import RedisCache

# Create a Redis cache instance
cache = RedisCache(
    host="localhost",
    port=6379,
    db=0,
    expiration_time=3600  # 1 hour
)

# Store data
cache.set("user_response", {"name": "John", "age": 30})

# Retrieve data
data = cache.get("user_response")
print(data)  # Output: {'name': 'John', 'age': 30}
```

### Use with OpenAI Client

```python
from datapizza.clients.openai import OpenAIClient
from datapizza.cache.redis import RedisCache

# Initialize cache
cache = RedisCache(host="localhost", port=6379, db=0)

# Create client with cache
client = OpenAIClient(api_key="YOUR_API_KEY", cache=cache)

# First call - cache MISS: OpenAI API is called, result is cached
response = client.invoke("What is Python?")
print(response.text)

# Second call with same input - cache HIT: returns cached result without calling OpenAI
response = client.invoke("What is Python?")
print(response.text)  # Same result, but retrieved from cache (much faster)
```

### Use with Agents

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.cache.redis import RedisCache
from datapizza.tools import tool

@tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

# Initialize cache
cache = RedisCache(host="localhost", port=6379)

# Create agent with cache
client = OpenAIClient(api_key="YOUR_API_KEY", cache=cache)
agent = Agent(name="assistant", client=client, tools=[get_weather])

# Responses will be cached automatically
response = agent.run("What is the weather in Rome?")
```

## Configuration

### Constructor Parameters

```python
RedisCache(
    host: str = "localhost",      # Redis server hostname
    port: int = 6379,             # Redis server port
    db: int = 0,                  # Redis database number (0-15)
    expiration_time: int = 3600   # Cache expiration in seconds (default: 1 hour)
)
```

### Parameters Explanation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | "localhost" | Hostname or IP address of Redis server |
| `port` | int | 6379 | Port number where Redis is listening |
| `db` | int | 0 | Redis database index (0-15). Use different DB numbers to partition cache data |
| `expiration_time` | int | 3600 | Time-to-live for cached entries in seconds |

### Environment Variables

For cloud deployments, configure Redis connection via environment variables:

```bash
REDIS_HOST=redis.example.com
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_password  # If Redis requires authentication
```

### Advanced Configuration

For production deployments with authentication:

```python
import redis
from datapizza.cache.redis import RedisCache

# Using SSL/TLS for remote Redis
cache = RedisCache(
    host="redis.prod.example.com",
    port=6380,
    db=0,
    expiration_time=7200
)
```

If you need custom Redis configuration, you can modify the `cache.py` implementation to accept additional parameters.

## Core Methods

### `get(key: str) -> object | None`

Retrieve a cached value by key.

```python
cache = RedisCache()

# Store data
cache.set("my_key", "my_value")

# Retrieve data
value = cache.get("my_key")
print(value)  # Output: "my_value"

# Non-existent key returns None
value = cache.get("non_existent_key")
print(value)  # Output: None
```

**Returns**: The cached object, or `None` if key doesn't exist or has expired.

### `set(key: str, obj: object) -> None`

Store a value in the cache with automatic expiration.

```python
cache = RedisCache(expiration_time=7200)  # 2 hours

# Store strings
cache.set("greeting", "Hello World")

# Store complex objects
cache.set("user_data", {
    "id": 123,
    "name": "Alice",
    "roles": ["admin", "user"]
})

# Store objects
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

user = User("Bob", "bob@example.com")
cache.set("current_user", user)
```

**Parameters**:
- `key`: Unique identifier for the cached item
- `obj`: Any serializable Python object


## Best Practices

### 1. **Connection Management**

```python
cache = RedisCache(host="localhost", port=6379)

client1 = OpenAIClient(api_key="KEY", cache=cache)
client2 = GoogleClient(api_key="KEY", cache=cache)
```

### 2. **Expiration Time**

```python
cache_minutes = RedisCache(expiration_time=300)  # 5 minutes

# Medium-lived cache (hourly updates)
cache_hours = RedisCache(expiration_time=3600)  # 1 hour

# Long-lived cache (stable data)
cache_days = RedisCache(expiration_time=86400)  # 1 day
```

### 3. **Key Naming Convention**

```python
# Good: Descriptive, hierarchical keys
cache.set("user:123:profile", user_data)
cache.set("product:456:details", product_info)
cache.set("ai:response:weather:Rome", response)

# Avoid: Generic or non-descriptive keys
cache.set("data", something)
cache.set("cache_1", value)
```

## Related Modules

- [datapizza-ai-core](../../../datapizza-ai-core) - Core framework
- [datapizza-ai-clients](../../datapizza-ai-clients) - LLM client adapters
- [datapizza-ai-cache](../) - Cache module (this package is a component)

## Roadmap
...


---
