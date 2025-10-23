# GoogleEmbedder

```python
pip install datapizza-ai-embedders-google
```

<!-- prettier-ignore -->
::: datapizza.embedders.google.GoogleEmbedder
    options:
        show_source: false


## Usage

```python
from datapizza.embedders.google import GoogleEmbedder

embedder = GoogleEmbedder(
    api_key="your-google-api-key"
)

# Embed a single text
embedding = embedder.embed("Hello world", model_name="models/embedding-001")

# Embed multiple texts
embeddings = embedder.embed(
    ["Hello world", "Another text"],
    model_name="models/embedding-001"
)
```

## Features

- Supports Google's Gemini embedding models
- Handles both single text and batch text embedding
- Async embedding support with `a_embed()`
- Automatic client initialization and management
- Uses Google's Generative AI SDK

## Examples

### Basic Text Embedding

```python
from datapizza.embedders.google import GoogleEmbedder

embedder = GoogleEmbedder(api_key="your-google-api-key")

# Single text embedding
text = "This is a sample document for embedding."
embedding = embedder.embed(text, model_name="models/embedding-001")

print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

### Async Embedding

```python
import asyncio
from datapizza.embedders.google import GoogleEmbedder

async def embed_async():
    embedder = GoogleEmbedder(api_key="your-google-api-key")

    text = "Async embedding example"
    embedding = await embedder.a_embed(text, model_name="models/embedding-001")

    return embedding

# Run async function
embedding = asyncio.run(embed_async())
```
