# DataPizza AI - Ollama Embedder

Ollama embedder implementation for DataPizza AI framework.

## Installation

```bash
pip install datapizza-ai-embedders-ollama
```

## Requirements

- Ollama must be installed and running locally
- Default port: `11434`
- Download the embedding model: `ollama pull qwen3-embedding:8b`

## Usage

### Basic Usage

```python
from datapizza.embedders.ollama import OllamaEmbedder

# Initialize with default settings
embedder = OllamaEmbedder()

# Embed a single text
embedding = embedder.embed("Hello, world!")
print(len(embedding))  # Vector dimension

# Embed multiple texts
embeddings = embedder.embed(["Hello", "World"])
print(len(embeddings))  # Number of texts
print(len(embeddings[0]))  # Vector dimension
```

### Custom Configuration

```python
# Use a different model
embedder = OllamaEmbedder(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434"
)

# Override model at runtime
embedding = embedder.embed(
    "Hello, world!",
    model_name="nomic-embed-text"
)
```

### Async Usage

```python
import asyncio
from datapizza.embedders.ollama import OllamaEmbedder

async def main():
    embedder = OllamaEmbedder()
    
    # Async embed
    embedding = await embedder.a_embed("Hello, world!")
    
    # Async embed multiple texts
    embeddings = await embedder.a_embed(["Hello", "World"])

asyncio.run(main())
```

## Parameters

- `model_name` (str, optional): Name of the Ollama embedding model. Default: `"qwen3-embedding:8b"`
- `base_url` (str, optional): Base URL for Ollama server. Default: `"http://localhost:11434"`

## Available Embedding Models

Popular Ollama embedding models:
- `qwen3-embedding:8b` (default)
- `mxbai-embed-large`
- `nomic-embed-text`
- `all-minilm`
- `snowflake-arctic-embed`

To pull a model:
```bash
ollama pull model-name
```

## Features

- ✅ Synchronous and asynchronous support
- ✅ Single and batch text embedding
- ✅ Local inference (no API keys needed)
- ✅ Multiple model support
- ✅ Custom server URL configuration

## License

MIT