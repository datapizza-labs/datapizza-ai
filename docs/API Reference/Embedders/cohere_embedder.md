# CohereEmbedder



```python
pip install datapizza-ai-embedders-cohere
```

<!-- prettier-ignore -->
::: datapizza.embedders.cohere.CohereEmbedder
    options:
        show_source: false


## Usage

```python
from datapizza.embedders.cohere import CohereEmbedder

embedder = CohereEmbedder(
    api_key="your-cohere-api-key",
    base_url="https://api.cohere.ai/v1",
    input_type="search_document"  # or "search_query"
)

# Embed a single text
embedding = embedder.embed("Hello world", model_name="embed-english-v3.0")

# Embed multiple texts
embeddings = embedder.embed(
    ["Hello world", "Another text"],
    model_name="embed-english-v3.0"
)
```

## Features

- Supports Cohere's embedding models
- Configurable input type for search documents or queries
- Handles both single text and batch text embedding
- Async embedding support with `a_embed()`
- Custom endpoint support for compatible APIs
- Uses Cohere's ClientV2 for optimal performance

## Examples

### Basic Text Embedding

```python
from datapizza.embedders.cohere import CohereEmbedder

embedder = CohereEmbedder(
    api_key="your-cohere-api-key",
    base_url="https://api.cohere.ai/v1",
    input_type="search_document"
)

# Single text embedding
text = "This is a sample document for embedding."
embedding = embedder.embed(text, model_name="embed-english-v3.0")

print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

### Search Query Embedding

```python
from datapizza.embedders.cohere import CohereEmbedder

# Configure for search queries
embedder = CohereEmbedder(
    api_key="your-cohere-api-key",
    base_url="https://api.cohere.ai/v1",
    input_type="search_query"
)

query = "What is machine learning?"
embedding = embedder.embed(query, model_name="embed-english-v3.0")

print(f"Query embedding size: {len(embedding)}")
```

### Batch Text Embedding

```python
from datapizza.embedders.cohere import CohereEmbedder

embedder = CohereEmbedder(
    api_key="your-cohere-api-key",
    base_url="https://api.cohere.ai/v1"
)

texts = [
    "First document to embed",
    "Second document to embed",
    "Third document to embed"
]

embeddings = embedder.embed(texts, model_name="embed-english-v3.0")

for i, emb in enumerate(embeddings):
    print(f"Document {i+1} embedding size: {len(emb)}")
```

### Async Embedding

```python
import asyncio
from datapizza.embedders.cohere import CohereEmbedder

async def embed_async():
    embedder = CohereEmbedder(
        api_key="your-cohere-api-key",
        base_url="https://api.cohere.ai/v1"
    )

    text = "Async embedding example"
    embedding = await embedder.a_embed(text, model_name="embed-english-v3.0")

    return embedding

# Run async function
embedding = asyncio.run(embed_async())
```
