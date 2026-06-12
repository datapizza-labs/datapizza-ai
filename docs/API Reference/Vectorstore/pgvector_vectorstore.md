# pgvector

```python
pip install datapizza-ai-vectorstores-pgvector
```

<!-- prettier-ignore -->
::: datapizza.vectorstores.pgvector.PgVectorVectorstore
    options:
        show_source: false

## Usage

```python
from datapizza.core.vectorstore import VectorConfig
from datapizza.vectorstores.pgvector import PgVectorVectorstore

store = PgVectorVectorstore(dsn="postgresql://user:pass@localhost:5432/db")

store.create_collection(
    "docs",
    vector_config=[VectorConfig(name="embedding", dimensions=1536)],
)

# Optional: create an ANN index immediately
store.create_collection(
    "docs_with_index",
    vector_config=[VectorConfig(name="embedding", dimensions=1536)],
    create_index=True,
)

# Query the collection
results = store.search(
    collection_name="docs",
    query_vector=[0.1, 0.2, ...],
    k=5,
)
```

## Advanced Usage

### Optional connection pooling

The vectorstore supports optional connection pooling using `psycopg_pool`. This is useful in multi-threaded or high-concurrency environments.

```python
store = PgVectorVectorstore(
    dsn="postgresql://user:pass@localhost:5432/db",
    use_pool=True,
    pool_min_size=2,
    pool_max_size=10,
)
```

> Note: `psycopg_pool` is optional. If `use_pool=True` and it is not installed, the constructor will raise an `ImportError`.

### JSONB filtering (metadata)

You can filter operations on the `metadata` column using JSONB containment. This works for `search`, `retrieve`, `update`, and `remove`.

```python
# Search only items with metadata.tag == "keep"
results = store.search(
    collection_name="docs",
    query_vector=[...],
    k=10,
    filters={"tag": "keep"},
)

# Update only filtered items
store.update(
    "docs",
    payload={"status": "updated"},
    points=[...],
    filters={"tag": "keep"},
)

# Remove only filtered items
store.remove(
    "docs",
    ids=[...],
    filters={"tag": "skip"},
)
```

## Performance & cleanup

### Batch inserts (executemany)

`add()` accepts either a single `Chunk` or a list of `Chunk` objects. When given a list, the implementation uses `executemany` for a batched insert, which is significantly faster for large uploads.

```python
chunks = [
    Chunk(
        id=str(uuid.uuid4()),
        text=f"batch {i}",
        embeddings=[DenseEmbedding(name="embedding", vector=[float(i), float(i)])],
    )
    for i in range(100)
]

store.add(chunks, collection_name="docs")
```

### Clean shutdown (pool close)

If you use connection pooling, it is recommended to close resources when your application shuts down.

```python
store.close()

# Async
await store.aclose()
```

## Collection management

```python
store.list_collections()
store.describe_collection("docs")
store.drop_collection("docs")
```

## Index management

```python
store.create_index(
    "docs",
    method="hnsw",
    index_name="docs_embedding_hnsw_idx",
)

store.list_indexes("docs")
store.describe_index("docs_embedding_hnsw_idx")
store.drop_index("docs_embedding_hnsw_idx")
```
