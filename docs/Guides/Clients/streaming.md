# Streaming

Streaming allows you to receive responses in real-time as they're generated, providing a better user experience for long responses and interactive applications.

## Why Use Streaming?

- **Real-time feedback**: Users see responses as they're generated
- **Better UX**: Reduces perceived latency for long responses
- **Progressive display**: Show partial results immediately
- **Interruptible**: Can stop generation early if needed

## Basic Streaming

### Synchronous Streaming

```python
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(
    api_key="your-api-key",
    model="gpt-4o-mini"
)

# Basic streaming
for chunk in client.stream_invoke("Write a short story about a robot learning to paint"):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
print()  # New line when complete
```

### Asynchronous Streaming

```python
import asyncio

async def async_stream_example():
    async for chunk in client.a_stream_invoke("Explain quantum computing in simple terms"):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
    print()  # New line when complete

# Run the async function
asyncio.run(async_stream_example())
```

## Streaming Structured Responses

Structured responses can also be streamed. Intermediate chunks carry the raw JSON text fragments in `chunk.delta`, while the final chunk carries the validated Pydantic instance in `chunk.structured_data`.

### Synchronous

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

final = None
for chunk in client.stream_structured_response(
    input="Create a profile for a software engineer",
    output_cls=Person,
):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)  # raw JSON fragments
    if chunk.partial:
        # best-effort dict of the JSON streamed so far, e.g. {'name': 'Ada'}
        name = chunk.partial.get("name")
    final = chunk

person = final.structured_data[0]
print(f"\nName: {person.name}, Age: {person.age}")
```

Every chunk also exposes:

- `chunk.snapshot` — the raw JSON text accumulated so far (the full text on the final chunk)
- `chunk.partial` — a best-effort parse of the snapshot (usually a dict), useful for progressive UIs. It is `None` until the buffer becomes parseable, and a field may still be incomplete until the following key appears in the stream (e.g. `{'age': 3}` before `{'age': 36}`). The final chunk's `structured_data` remains the only validated result.

### Asynchronous

```python
import asyncio

async def async_structured_stream_example():
    final = None
    async for chunk in client.a_stream_structured_response(
        input="Create a profile for a software engineer",
        output_cls=Person,
    ):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
        final = chunk
    person = final.structured_data[0]
    print(f"\nName: {person.name}, Age: {person.age}")

asyncio.run(async_structured_stream_example())
```

If the stream does not produce valid JSON for `output_cls`, a `pydantic.ValidationError` is raised when the final chunk is built. Clients that do not support streaming structured responses raise `NotImplementedError` on first iteration.
