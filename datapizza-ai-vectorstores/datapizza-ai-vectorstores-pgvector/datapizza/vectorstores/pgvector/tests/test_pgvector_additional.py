import asyncio
import uuid

import pytest
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import Chunk, DenseEmbedding

from datapizza.vectorstores.pgvector import PgVectorVectorstore


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_async_add_search(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "async_collection",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    chunk_id = str(uuid.uuid4())
    await store.a_add(
        Chunk(
            id=chunk_id,
            text="async",
            embeddings=[DenseEmbedding(name="embedding", vector=[0.0, 0.0, 0.0, 0.0])],
        ),
        collection_name="async_collection",
    )

    results = await store.a_search(
        collection_name="async_collection",
        query_vector=[0.0, 0.0, 0.0, 0.0],
        k=1,
    )

    assert len(results) == 1
    assert results[0].id == chunk_id


@pytest.mark.integration
def test_pgvector_update_multiple_records(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "update_batch",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    ids = [str(uuid.uuid4()) for _ in range(2)]
    for i, chunk_id in enumerate(ids, start=1):
        store.add(
            Chunk(
                id=chunk_id,
                text=f"row{i}",
                embeddings=[DenseEmbedding(name="embedding", vector=[0.0, 0.0, 0.0, 0.0])],
                metadata={"tag": f"initial{i}"},
            ),
            collection_name="update_batch",
        )

    store.update(
        collection_name="update_batch",
        payload={"updated": True},
        points=ids,
    )

    results = store.retrieve("update_batch", ids=ids)
    assert len(results) == 2
    for r in results:
        assert r.metadata.get("updated") is True
        assert r.metadata.get("tag") is not None


@pytest.mark.unit
def test_pgvector_search_in_empty_cache_raises():
    store = PgVectorVectorstore(dsn="postgresql://localhost:5432/test")
    # Emulate case where create_collection was never called / cache not populated
    store._collections.clear()

    with pytest.raises(ValueError):
        store.search(
            collection_name="missing", query_vector=[0.0, 0.0, 0.0, 0.0], k=1
        )


@pytest.mark.integration
def test_pgvector_cache_rebuild_from_schema(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "cache_rebuild",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    # Clear the cache and ensure search still works by rebuilding from schema
    store._collections.clear()
    results = store.search(
        collection_name="cache_rebuild",
        query_vector=[0.0, 0.0, 0.0, 0.0],
        k=1,
    )

    assert isinstance(results, list)
    assert "cache_rebuild" in store._collections


@pytest.mark.integration
def test_pgvector_drop_collection(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "to_drop",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    store.drop_collection("to_drop")

    with pytest.raises(Exception):
        store.search(
            collection_name="to_drop", query_vector=[0.0, 0.0, 0.0, 0.0], k=1
        )


@pytest.mark.integration
def test_pgvector_create_index(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "with_index",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    store.create_index("with_index", method="hnsw")

    conn = store._get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname=%s AND tablename=%s;",
            (store.schema, "with_index"),
        )
        indexes = [r[0] for r in cur.fetchall()]

    assert any("with_index" in idx for idx in indexes)


@pytest.mark.integration
def test_pgvector_context_manager_closes(pgvector_postgres):
    with PgVectorVectorstore(dsn=pgvector_postgres) as store:
        store.create_collection(
            "cm",
            vector_config=[VectorConfig(name="embedding", dimensions=4)],
        )
    assert store._conn is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_async_context_manager_closes(pgvector_postgres):
    async with PgVectorVectorstore(dsn=pgvector_postgres) as store:
        store.create_collection(
            "cm_async",
            vector_config=[VectorConfig(name="embedding", dimensions=4)],
        )
    assert store._a_conn is None


@pytest.mark.integration
def test_pgvector_list_and_describe_collections(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "list_test",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    collections = store.list_collections()
    assert "list_test" in collections

    desc = store.describe_collection("list_test")
    assert desc["name"] == "list_test"
    assert desc["schema"] == store.schema
    assert "vectors" in desc and len(desc["vectors"]) == 1
    assert desc["vectors"][0]["name"] == "embedding"
    assert isinstance(desc["vectors"][0]["dimensions"], (int, type(None)))


@pytest.mark.integration
def test_pgvector_index_listing_and_drop(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "index_test",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    # Create a named index
    store.create_index(
        "index_test",
        method="hnsw",
        index_name="idx_index_test_embedding_hnsw",
    )

    indexes = store.list_indexes("index_test")
    assert any(i["name"] == "idx_index_test_embedding_hnsw" for i in indexes)

    info = store.describe_index("idx_index_test_embedding_hnsw")
    assert info["table"] == "index_test"

    store.drop_index("idx_index_test_embedding_hnsw")
    indexes_after = store.list_indexes("index_test")
    assert all(i["name"] != "idx_index_test_embedding_hnsw" for i in indexes_after)


@pytest.mark.integration
@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_async_index_ivfflat(pgvector_postgres):
    async with PgVectorVectorstore(dsn=pgvector_postgres) as store:
        store.create_collection(
            "index_test_ivfflat_async",
            vector_config=[VectorConfig(name="embedding", dimensions=4)],
        )

        await store.a_create_index(
            "index_test_ivfflat_async",
            method="ivfflat",
            ivfflat_lists=8,
            index_name="idx_index_test_ivfflat_async",
        )

        indexes = store.list_indexes("index_test_ivfflat_async")
        assert any(i["name"] == "idx_index_test_ivfflat_async" for i in indexes)

        info = store.describe_index("idx_index_test_ivfflat_async")
        assert info["table"] == "index_test_ivfflat_async"
        assert info.get("method") == "ivfflat"
        assert info.get("ivfflat_lists") == 8

        await store.a_drop_index("idx_index_test_ivfflat_async")
        indexes_after = store.list_indexes("index_test_ivfflat_async")
        assert all(i["name"] != "idx_index_test_ivfflat_async" for i in indexes_after)


@pytest.mark.integration
def test_pgvector_full_integration_flow(pgvector_postgres):
    """Full integration test: CRUD, search, index, list/describe, drop."""
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "integration_smoke",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
        create_index=True,
    )

    store.add(
        Chunk(
            id="smoke-1",
            text="hello",
            embeddings=[DenseEmbedding(name="embedding", vector=[0.1, 0.1, 0.1, 0.1])],
        ),
        collection_name="integration_smoke",
    )

    results = store.search(
        collection_name="integration_smoke",
        query_vector=[0.1, 0.1, 0.1, 0.1],
        k=1,
    )
    assert len(results) == 1
    assert results[0].id == "smoke-1"

    assert "integration_smoke" in store.list_collections()

    # Choose a non-pkey index to drop
    indexes = store.list_indexes("integration_smoke")
    idx_name = next(i["name"] for i in indexes if not i["name"].endswith("_pkey"))

    info = store.describe_index(idx_name)
    assert info["table"] == "integration_smoke"

    store.drop_index(idx_name)
    store.drop_collection("integration_smoke")
