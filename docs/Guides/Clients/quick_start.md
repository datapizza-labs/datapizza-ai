# Quick Start

This guide will help you get started with the `OpenAIClient` in datapizza-ai. For specialized topics, check out our detailed guides on [multimodality](multimodality.md), [streaming](streaming.md) and [building chatbots](chatbot.md).

## Installation

First, make sure you have datapizza-ai installed:

```bash
pip install datapizza-ai
```

## Basic Setup


```python
from datapizza.clients.openai import OpenAIClient

# Initialize the client with your API key
client = OpenAIClient(
    api_key="your-openai-api-key",
    model="gpt-4o-mini",  # Default model
    system_prompt="You are a helpful assistant",  # Optional
    temperature=0.7  # Optional, controls randomness (0-2)
)
```


```python
# Basic text response
response = client.invoke("What is the capital of France?")
print(response.text)
# Output: "The capital of France is Paris."
```

## Core Methods


```python
response = client.invoke(
    input="Explain quantum computing in simple terms",
    temperature=0.5,  # Override default temperature
    max_tokens=200,   # Limit response length
    system_prompt="You are a physics teacher"  # Override system prompt
)

print(response.text)
print(f"Tokens used: {response.completion_tokens_used}")
```


## Async invoke

```python
import asyncio

async def main():
    return await client.a_invoke(
        input="Explain quantum computing in simple terms",
        temperature=0.5,  # Override default temperature
        max_tokens=200,   # Limit response length
        system_prompt="You are a physics teacher"  # Override system prompt
    )

response = asyncio.run(main())

print(response.text)
print(f"Tokens used: {response.completion_tokens_used}")
```
## Working with Memory

Memory allows you to maintain conversation context:

```python
from datapizza.memory import Memory
from datapizza.type import ROLE, TextBlock

memory = Memory()

# First interaction
response1 = client.invoke("My name is Alice", memory=memory)
memory.add_turn(TextBlock(content="My name is Alice"), role=ROLE.USER)
memory.add_turn(response1.content, role=ROLE.ASSISTANT)

# Second interaction - the model remembers Alice
response2 = client.invoke("What's my name?", memory=memory)
print(response2.text)  # Should mention Alice
```

## Token Management
Monitor your usage:

```python
response = client.invoke("Explain AI")
print(f"Tokens used: {response.completion_tokens_used}")
print(f"Prompt token used: {response.prompt_tokens_used}")
print(f"Cached token used: {response.cached_tokens_used}")
```

That's it! You're ready to start building with the OpenAI client. Check out the specialized guides above for advanced features and patterns.



## What's Next?

Now that you know the basics, explore our specialized guides:

### 📸 [Multimodality Guide](multimodality.md)
Work with images, PDFs, and other media types for visual AI applications.

### 🌊 [Streaming Guide](streaming.md)
Build responsive applications with real-time text generation and streaming.

### 🛠️ [Tools Guide](tools.md)
Extend AI capabilities by integrating external functions and tools.

### 📊 [Structured Responses Guide](structured_responses.md)
Work with strongly-typed outputs using JSON schemas and Pydantic models.

### 🤖 [Chatbot Guide](chatbot.md)
Create sophisticated conversational AI with memory and context management.

