# DataPizza AI - OpenAI-Like Client

A versatile client for DataPizza AI that supports OpenAI-compatible APIs, including local models through Ollama, Together AI, and other OpenAI-compatible services.

## Installation

```bash
pip install datapizza-ai-clients-openai-like
```

## Quick Start

### With Ollama (Local Models)

```python
from datapizza.clients.openai_like import OpenAILikeClient

# Create client for Ollama
client = OpenAILikeClient(
    api_key="",  # Ollama doesn't require an API key
    model="gemma2:2b",
    system_prompt="You are a helpful assistant.",
    base_url="http://localhost:11434/v1",
)

response = client.invoke("What is the capital of France?")
print(response.content)
```

### With Together AI

```python
import os
from datapizza.clients.openai_like import OpenAILikeClient

client = OpenAILikeClient(
    api_key=os.getenv("TOGETHER_API_KEY"),
    model="meta-llama/Llama-2-7b-chat-hf",
    system_prompt="You are a helpful assistant.",
    base_url="https://api.together.xyz/v1",
)

response = client.invoke("Explain quantum computing")
print(response.content)
```

### With OpenRouter

```python
import os
from datapizza.clients.openai_like import OpenAILikeClient

client = OpenAILikeClient(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemma-7b-it",
    system_prompt="You are a helpful assistant.",
    base_url="https://openrouter.ai/api/v1",
)

response = client.invoke("What is OpenRouter?")
print(response.content)
```

### With Other OpenAI-Compatible Services

```python
import os
from datapizza.clients.openai_like import OpenAILikeClient

client = OpenAILikeClient(
    api_key=os.getenv("YOUR_API_KEY"),
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
    base_url="https://your-service-url/v1",
)

response = client.invoke("Your question here")
print(response.content)
```

## Features

- **OpenAI-Compatible**: Works with any service that implements the OpenAI API standard
- **Local Models**: Perfect for running with Ollama for privacy and cost control
- **Memory Support**: Built-in memory adapter for conversation history
- **Streaming**: Support for real-time streaming responses
- **Structured Outputs**: Generate structured data with Pydantic models
- **Tool Calling**: Function calling capabilities where supported

## Supported Services

- **Ollama** - Local model inference
- **Together AI** - Cloud-based model hosting
- **OpenRouter** - Access a variety of models through a single API
- **Perplexity AI** - Search-augmented models
- **Groq** - Fast inference API
- **Any OpenAI-compatible API**

## Advanced Usage

### With Memory

```python
from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.memory import Memory

client = OpenAILikeClient(
    api_key="",
    model="llama3.1:8b",
    base_url="http://localhost:11434/v1",
)

memory = Memory(client=client)
memory.add("I'm working on a Python project about machine learning.")
response = memory.query("What libraries should I use?")
```

### Streaming Responses

```python
client = OpenAILikeClient(
    api_key="",
    model="gemma2:7b",
    base_url="http://localhost:11434/v1",
)

for chunk in client.stream("Tell me a story about AI"):
    print(chunk.content, end="", flush=True)
```

### Structured Outputs

```python
from pydantic import BaseModel
from datapizza.clients.openai_like import OpenAILikeClient

class Person(BaseModel):
    name: str
    age: int
    occupation: str

client = OpenAILikeClient(
    api_key="",
    model="llama3.1:8b",
    base_url="http://localhost:11434/v1",
)

response = client.invoke(
    "Generate a person profile",
    response_format=Person
)
print(response.parsed)  # Person object
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api_key` | API key for the service | Required (empty string for Ollama) |
| `model` | Model name to use | Required |
| `base_url` | Base URL for the API | Required |
| `system_prompt` | System message for the model | None |
| `temperature` | Sampling temperature (0-2) | 0.7 |
| `max_tokens` | Maximum tokens in response | None |
| `timeout` | Request timeout in seconds | 30 |

## Ollama Setup

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull gemma2:2b`
3. Start Ollama: `ollama serve`
4. Use with DataPizza AI as shown in the examples above

## Popular Ollama Models

- `gemma2:2b` - Lightweight, fast responses
- `gemma2:7b` - Balanced performance
- `llama3.1:8b` - High quality, more resource intensive
- `codellama:7b` - Specialized for coding tasks
- `mistral:7b` - Good general purpose model

## Error Handling

```python
from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.core.clients.exceptions import ClientError

try:
    client = OpenAILikeClient(
        api_key="",
        model="nonexistent-model",
        base_url="http://localhost:11434/v1",
    )
    response = client.invoke("Hello")
except ClientError as e:
    print(f"Client error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.
