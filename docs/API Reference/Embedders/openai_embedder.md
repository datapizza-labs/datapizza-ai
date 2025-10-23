# OpenAIEmbedder

<!-- prettier-ignore -->
::: datapizza.embedders.openai.OpenAIEmbedder
    options:
        show_source: false


## Usage

```python
from datapizza.embedders.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(
    api_key="your-openai-api-key",
    base_url="https://api.openai.com/v1"  # Optional custom base URL
)

# Embed a single text
embedding = embedder.embed("Hello world", model_name="text-embedding-ada-002")

# Embed multiple texts
embeddings = embedder.embed(
    ["Hello world", "Another text"],
    model_name="text-embedding-ada-002"
)
```

## Features

- Supports OpenAI's embedding models
- Handles both single text and batch text embedding
- Async embedding support with `a_embed()`
- Custom base URL support for compatible APIs
- Automatic client initialization and management

## Examples

### Basic Text Embedding

```python
from datapizza.embedders.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(api_key="your-api-key")

# Single text embedding
text = "This is a sample document for embedding."
embedding = embedder.embed(text, model_name="text-embedding-ada-002")

print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

### Async Embedding

```python
import asyncio
from datapizza.embedders.openai import OpenAIEmbedder

async def embed_async():
    embedder = OpenAIEmbedder(api_key="your-api-key")

    text = "Async embedding example"
    embedding = await embedder.a_embed(text, model_name="text-embedding-ada-002")

    return embedding

# Run async function
embedding = asyncio.run(embed_async())
```
