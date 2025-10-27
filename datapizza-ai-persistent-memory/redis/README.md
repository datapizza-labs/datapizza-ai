<div align="center">
<img src="https://github.com/datapizza-labs/datapizza-ai/raw/main/docs/assets/logo_bg_dark.png" alt="Datapizza AI Logo" width="200" height="200">

# Datapizza AI - Redis Memory

**A Redis-based memory implementation for Datapizza AI that allows agents to store and retrieve chat session messages
efficiently.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

This submodule provides a Redis-based memory implementation for `datapizza-ai`. It extends the original Memory class to
store chat session messages in Redis, allowing for efficient and persistent retrieval and management of conversation
turns.

## Installation

To install the required dependencies, run:

```bash
# Install the core framework
pip install datapizza-ai
# Install the Redis Memory
pip install datapizza-ai-persistent-memory-redis
```

## Usage

### Initialization

```python
from datapizza.persistent_memory.redis import RedisMemory

# Initialize RedisMemory with user_id and session_id
memory = RedisMemory(user_id="unique_id_for_user", session_id="unique_id_for_chat_session")
```

### Adding Turns

```python
from datapizza.type import ROLE, TextBlock

# Add a new turn with a TextBlock
memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
# Add a new turn with multiple blocks
memory.add_turn(blocks=[TextBlock(content="How are you?"), TextBlock(content="I'm fine, thank you.")],
                role=ROLE.ASSISTANT)
```

### Retrieving Messages

```python
# Fetch all session messages
messages = memory.fetch_session_messages()

# Print messages
for message in messages:
    print(message)
```

### Clearing Memory

```python
# Clear all stored messages
memory.clear()
```

