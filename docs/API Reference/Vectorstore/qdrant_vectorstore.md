# Qdrant

```python
pip install datapizza-ai-vectorstores-qdrant
```

<!-- prettier-ignore -->
::: datapizza.vectorstores.qdrant.QdrantVectorstore
    options:
        show_source: false


## Usage

```python
from datapizza.vectorstores.qdrant import QdrantVectorstore

# Connect to Qdrant server
vectorstore = QdrantVectorstore(
    host="localhost",
    port=6333,
    api_key="your-api-key"  # Optional
)

# Or use in-memory/file storage
vectorstore = QdrantVectorstore(
    location=":memory:"  # Or path to file
)
```

## Features

- Connect to Qdrant server or use local storage
- Support for both dense and sparse embeddings
- Named vector configurations for multi-vector collections
- Batch operations for efficient processing
- Collection management (create, delete, list)
- Chunk-based operations with metadata preservation
- Async support for all operations
- Point-level operations (add, update, remove, retrieve)

## Examples

### Basic Setup and Collection Creation

```python
from datapizza.core.vectorstore import Distance, VectorConfig
from datapizza.type import EmbeddingFormat
from datapizza.vectorstores.qdrant import QdrantVectorstore

vectorstore = QdrantVectorstore(location=":memory:")

# Create collection with vector configuration
vector_config = [
    VectorConfig(
        name="text_embeddings",
        dimensions=3,
        format=EmbeddingFormat.DENSE,
        distance=Distance.COSINE
    )
]

vectorstore.create_collection(
    collection_name="documents",
    vector_config=vector_config
)

# Add nodes and search

import uuid
from datapizza.type import Chunk, DenseEmbedding
from datapizza.vectorstores.qdrant import QdrantVectorstore

# Create chunks with embeddings
chunks = [
    Chunk(
        id=str(uuid.uuid4()),
        text="First document content",
        metadata={"source": "doc1.txt"},
        embeddings=[DenseEmbedding(name="text_embeddings", vector=[0.1, 0.2, 0.3])]
    ),
    Chunk(
        id=str(uuid.uuid4()),
        text="Second document content",
        metadata={"source": "doc2.txt"},
        embeddings=[DenseEmbedding(name="text_embeddings", vector=[0.4, 0.5, 0.6])]
    )
]

# Add chunks to collection
vectorstore.add(chunks, collection_name="documents")

# Search for similar chunks
query_vector = [0.1, 0.2, 0.3]
results = vectorstore.search(
    collection_name="documents",
    query_vector=query_vector,
    k=5
)

for chunk in results:
    print(f"Text: {chunk.text}")
    print(f"Metadata: {chunk.metadata}")
```
