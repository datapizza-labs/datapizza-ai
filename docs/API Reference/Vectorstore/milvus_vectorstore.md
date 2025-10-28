# Milvus



```bash
pip install datapizza-ai-vectorstores-milvus
```

<!-- prettier-ignore
::: datapizza.vectorstores.milvus.MilvusVectorstore
    options:
        show_source: false

 -->


## Usage
```python
from datapizza.vectorstores.milvus import MilvusVectorstore

# Option A) Milvus Server / Zilliz Cloud
vectorstore = MilvusVectorstore(
    host="localhost",
    port=19530,
    # user="username",            # Optional
    # password="password",        # Optional
    # secure=True,                # Optional (TLS)
    # token="zilliz_token",       # Optional (Zilliz)
)

# Option B) Single-URI style (works with Milvus, Zilliz, or Milvus Lite)
vectorstore = MilvusVectorstore(uri="./milvus.db")  # Milvus Lite

```

## Features

- Connect via `host/port` or a single `uri` (supports Milvus Server, Zilliz Cloud, and Milvus Lite).
- Works with **dense** and **sparse** embeddings in the *same* collection.
- Named vector fields for **multi-vector** collections.
- Batch/async operations: `add` / `a_add`, `search` / `a_search`.
- Collection management: `create_collection`, `delete_collection`, `get_collections`.
- Entities ops: `retrieve`, `update` (upsert), `remove`.
- Flexible indexing (defaults provided; accepts custom `IndexParams`).
- Dynamic metadata via Milvus’ `$meta` (stored from `Chunk.metadata`).

---

## Examples

### Basic Setup & Collection Creation

```python
import uuid
from datapizza.core.vectorstore import Distance, VectorConfig
from datapizza.type import EmbeddingFormat
from datapizza.vectorstores.milvus import MilvusVectorstore


# Local Lite (file) for quick dev
vectorstore = MilvusVectorstore(uri="./milvus.db")

# Declare your vector fields (can be multiple)
vector_config = [
    VectorConfig(
        name="text_embeddings",
        dimensions=3,
        format=EmbeddingFormat.DENSE,
        distance=Distance.COSINE,
    )
]

# Create (id, text, and above vectors; dynamic metadata enabled)
vectorstore.create_collection(
    collection_name="documents",
    vector_config=vector_config
)

```



### Add Chunks & Search

```python
from datapizza.type import Chunk, DenseEmbedding

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
    ),
]

vectorstore.add(chunks, collection_name="documents")

# IMPORTANT: provide the vector field name if not using an Embedding instance
results = vectorstore.search(
    collection_name="documents",
    query_vector=[0.1, 0.2, 0.3],
    vector_name="text_embeddings",
    k=5
)

for chunk in results:
    print(chunk.id, chunk.text, chunk.metadata)

```



### Retrieve, Update (Upsert), Remove

```python
# Suppose you kept an id from earlier
target_id = chunks[0].id

# Retrieve by ids
found = vectorstore.retrieve(collection_name="documents", ids=[target_id])
print("Retrieved:", found[0].text)

updated = Chunk(
    id=target_id,
    text="First document content (updated)",
    metadata={"source": "doc1.txt", "version": 2},
    embeddings=[DenseEmbedding(name="text_embeddings", vector=[0.11, 0.19, 0.31])],
)

vectorstore.update(collection_name="documents", chunk=updated)

vectorstore.remove(collection_name="documents", ids=[target_id])

```



### Async API

```python
import asyncio
from datapizza.type import DenseEmbedding

async def main():
    vs = MilvusVectorstore(uri="./milvus.db")
    more = [
        Chunk(
            id=str(uuid.uuid4()),
            text="Async doc",
            embeddings=[DenseEmbedding(name="text_embeddings", vector=[0.2, 0.1, 0.9])]
        )
    ]
    await vs.a_add(more, collection_name="documents")

    hits = await vs.a_search(
        collection_name="documents",
        query_vector=DenseEmbedding(name="text_embeddings", vector=[0.2, 0.1, 0.9]),
        k=3,
    )
    print([h.text for h in hits])

asyncio.run(main())
```

### Multi-Vector (Dense + Sparse)

```python
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import Chunk, DenseEmbedding, SparseEmbedding, EmbeddingFormat
from datapizza.vectorstores.milvus import MilvusVectorstore
import uuid

vectorstore = MilvusVectorstore()
# You can also specify advanced IndexParams
index_params = vectorstore.prepare_index_params()

index_params.add_index(
    field_name="dense_vector", # Name of the vector field to be indexed
    index_type="GPU_IVF_FLAT", # Type of the index to create
    index_name="dense_vector_index", # Name of the index to create
    metric_type="L2", # Metric type used to measure similarity
    params={
        "nlist": 1024, # Number of clusters for the index
    } # Index building params
)

index_params.add_index(
    field_name="sparse_vector", # Name of the vector field to be indexed
    index_type="SPARSE_INVERTED_INDEX", # Type of the index to create
    index_name="sparse_vector_index", # Name of the index to create
    metric_type="IP", # Metric type used to measure similarity
    params={"inverted_index_algo": "DAAT_MAXSCORE"}, # Algorithm used for building and querying the index
)
# Create with dense + sparse vector fields

vectorstore.create_collection(
    collection_name="hybrid_docs",
    vector_config=[
        VectorConfig(
            name="dense_vector",
            dimensions=1024,
            format=EmbeddingFormat.DENSE,
        ),
        VectorConfig(
            name="sparse_vector",
            dimensions=0,  # ignored by Milvus for sparse
            format=EmbeddingFormat.SPARSE,
        ),
    ],
    index_params=index_params  # index params created earlier
)

hybrid = Chunk(
    id=str(uuid.uuid4()),
    text="Hybrid vector example",
    metadata={"lang": "en"},
    embeddings=[
        DenseEmbedding(name="dense_vector", vector=[0.01]*1024),
        SparseEmbedding(name="sparse_vector", indices=[1, 7, 42], values=[0.9, 0.5, 0.2]),
    ],
)

vectorstore.add(hybrid, collection_name="hybrid_docs")

# Dense search
vectorstore.search(
    collection_name="hybrid_docs",
    query_vector=DenseEmbedding(name="dense_vector", vector=[0.01]*1024),
    k=5,
)

# Sparse search
vectorstore.search(
    collection_name="hybrid_docs",
    query_vector=SparseEmbedding(name="sparse_vector", indices=[1,7], values=[1.0,0.6]),
    k=5,
)
```


### Dump a Collection (Pagination Helper

```python
for chunk in vectorstore.dump_collection("documents", page_size=100):
    print(chunk.id, chunk.text, chunk.metadata)
```
