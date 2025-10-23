# ChunkEmbedder

<!-- prettier-ignore -->
::: datapizza.embedders.ChunkEmbedder
    options:
        show_source: false


## Usage

```python
from datapizza.embedders import ChunkEmbedder
from datapizza.core.clients import Client

# Initialize with any compatible client
client = Client(...)  # Your client instance
embedder = ChunkEmbedder(
    client=client,
    model_name="text-embedding-ada-002",  # Optional model override
    embedding_name="my_embeddings",       # Optional custom embedding name
    batch_size=100                        # Optional batch size for processing
)

# Embed chunks - adds embeddings to chunk objects
embedded_chunks = embedder.embed(chunks)
```

## Features

- Specialized for embedding lists of Chunk objects
- Batch processing with configurable batch size
- Adds embeddings directly to Chunk objects
- Preserves original chunk structure and metadata
- Async embedding support with `a_embed()`
- Memory efficient batch processing
- Works with any compatible LLM client

## Examples

### Basic Chunk Embedding

```python
import os

from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.type import Chunk
from dotenv import load_dotenv

load_dotenv()

# Create client and embedder
client = OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY"))
embedder = ChunkEmbedder(
    client=client,
    model_name="text-embedding-ada-002",
    batch_size=50
)

# Create sample chunks
chunks = [
    Chunk(id="1", text="First chunk of text", metadata={"source": "doc1"}),
    Chunk(id="2", text="Second chunk of text", metadata={"source": "doc2"}),
    Chunk(id="3", text="Third chunk of text", metadata={"source": "doc3"})
]

# Embed chunks (modifies chunks in-place)
embedded_chunks = embedder.embed(chunks)

# Check embeddings were added
for i, chunk in enumerate(embedded_chunks):
    print(f"Chunk {i+1}:")
    print(f"  Text: {chunk.text[:50]}...")
    print(f"  Embeddings: {len(chunk.embeddings)}")
    if chunk.embeddings:
        print(f"  Embedding name: {chunk.embeddings[0].name}")
        print(f"  Vector size: {len(chunk.embeddings[0].vector)}")
```
