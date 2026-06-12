# datapizza-ai-vectorstores-pgvector

PostgreSQL + pgvector vectorstore implementation for the datapizza-ai framework.

This package provides a `PgVectorVectorstore` that implements the `Vectorstore` interface from `datapizza-ai-core`.

## Quickstart

```python
from datapizza.core.vectorstore import VectorConfig
from datapizza.vectorstores.pgvector import PgVectorVectorstore

store = PgVectorVectorstore(dsn="postgresql://user:pass@localhost:5432/db")

# Create a collection (table) with a vector column
store.create_collection("docs", vector_config=[VectorConfig(name="embedding", dimensions=1536)])

# Optional: create an ANN index immediately (HNSW by default)
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

# Drop the collection (deletes the table)
store.drop_collection("docs")

# Inspect existing collections
print(store.list_collections())
print(store.describe_collection("docs"))
```

## Advanced Usage

### Optional connection pooling (enterprise)

`PgVectorVectorstore` supports optional connection pooling using `psycopg_pool`.

```python
from datapizza.vectorstores.pgvector import PgVectorVectorstore

store = PgVectorVectorstore(
    dsn="postgresql://user:pass@localhost:5432/db",
    use_pool=True,
    pool_min_size=2,
    pool_max_size=10,
)
```

> Note: `psycopg_pool` is optional. If `use_pool=True` and it is not installed, the constructor will raise an `ImportError`.

### Metadata filtering (JSONB)

You can filter search/update/remove/retrieve operations using JSONB filtering on the `metadata` column.

```python
# Only search in items where metadata contains {"tag": "keep"}
results = store.search(
    collection_name="docs",
    query_vector=[...],
    k=10,
    filters={"tag": "keep"},
)

# Update only items matching a filter
store.update(
    "docs",
    payload={"status": "updated"},
    points=[...],
    filters={"tag": "keep"},
)

# Remove items matching a filter
store.remove(
    "docs",
    ids=[...],
    filters={"tag": "skip"},
)
```

## Performance & cleanup

### Batch inserts (executemany)

`add()` accepts a single `Chunk` or a list of `Chunk` objects. When passing a list, the store performs a batched insert using `executemany`, which is much faster for large uploads.

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

If you enable pooling, it's a good idea to explicitly close resources when your application shuts down.

```python
store.close()

# Async
await store.aclose()
```
```
