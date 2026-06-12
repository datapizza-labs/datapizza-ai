import uuid

import pytest
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import Chunk, DenseEmbedding

from datapizza.vectorstores.pgvector import PgVectorVectorstore


@pytest.mark.integration
def test_pgvector_collection_add_search(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "test_collection",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    chunk_id = str(uuid.uuid4())
    store.add(
        Chunk(
            id=chunk_id,
            text="hello",
            embeddings=[DenseEmbedding(name="embedding", vector=[0.0, 0.0, 0.0, 0.0])],
            metadata={"tag": "keep"},
        ),
        collection_name="test_collection",
    )

    # Add a second chunk with different metadata to ensure filters work.
    store.add(
        Chunk(
            id=str(uuid.uuid4()),
            text="other",
            embeddings=[DenseEmbedding(name="embedding", vector=[0.0, 0.0, 0.0, 0.0])],
            metadata={"tag": "skip"},
        ),
        collection_name="test_collection",
    )

    results = store.search(
        collection_name="test_collection",
        query_vector=[0.0, 0.0, 0.0, 0.0],
        k=10,
        filters={"tag": "keep"},
    )

    assert len(results) == 1
    assert results[0].id == chunk_id
    assert results[0].text == "hello"


@pytest.mark.integration
def test_pgvector_collection_create_collection_schema_mismatch_raises(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "test_collection_mismatch",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    with pytest.raises(ValueError):
        store.create_collection(
            "test_collection_mismatch",
            vector_config=[VectorConfig(name="embedding", dimensions=8)],
        )


@pytest.mark.integration
def test_pgvector_collection_pooling_option(pgvector_postgres):
    pytest.importorskip("psycopg_pool")

    store = PgVectorVectorstore(dsn=pgvector_postgres, use_pool=True)

    store.create_collection(
        "test_collection_pool",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    chunk_id = str(uuid.uuid4())
    store.add(
        Chunk(
            id=chunk_id,
            text="pool-test",
            embeddings=[DenseEmbedding(name="embedding", vector=[0.0, 0.0, 0.0, 0.0])],
        ),
        collection_name="test_collection_pool",
    )

    results = store.search(
        collection_name="test_collection_pool",
        query_vector=[0.0, 0.0, 0.0, 0.0],
        k=1,
    )

    assert len(results) == 1
    assert results[0].id == chunk_id


def test_pgvector_collection_remove_and_retrieve(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "test_collection_2",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    chunk_id = str(uuid.uuid4())
    store.add(
        Chunk(
            id=chunk_id,
            text="remove-me",
            embeddings=[DenseEmbedding(name="embedding", vector=[0.0, 0.0, 0.0, 0.0])],
        ),
        collection_name="test_collection_2",
    )

    # Verify retrieval works
    retrieved = store.retrieve("test_collection_2", ids=[chunk_id])
    assert len(retrieved) == 1
    assert retrieved[0].id == chunk_id

    # Remove and verify gone
    store.remove("test_collection_2", ids=[chunk_id])
    retrieved_after = store.retrieve("test_collection_2", ids=[chunk_id])
    assert retrieved_after == []


def test_pgvector_update_and_remove_with_filters(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "test_collection_filters",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    # Add three chunks with varying metadata
    chunk_a = Chunk(
        id=str(uuid.uuid4()),
        text="a",
        embeddings=[DenseEmbedding(name="embedding", vector=[0.0, 0.0, 0.0, 0.0])],
        metadata={"tag": "keep", "group": "x"},
    )
    chunk_b = Chunk(
        id=str(uuid.uuid4()),
        text="b",
        embeddings=[DenseEmbedding(name="embedding", vector=[0.0, 0.0, 0.0, 0.0])],
        metadata={"tag": "keep", "group": "y"},
    )
    chunk_c = Chunk(
        id=str(uuid.uuid4()),
        text="c",
        embeddings=[DenseEmbedding(name="embedding", vector=[0.0, 0.0, 0.0, 0.0])],
        metadata={"tag": "skip", "group": "x"},
    )

    store.add(chunk_a, collection_name="test_collection_filters")
    store.add(chunk_b, collection_name="test_collection_filters")
    store.add(chunk_c, collection_name="test_collection_filters")

    # Update only the "keep" group items to add a "status" field.
    store.update(
        "test_collection_filters",
        payload={"status": "updated"},
        points=[chunk_a.id, chunk_b.id, chunk_c.id],
        filters={"tag": "keep"},
    )

    # Only the two matching items should have the new field.
    results = store.search(
        collection_name="test_collection_filters",
        query_vector=[0.0, 0.0, 0.0, 0.0],
        k=10,
        filters={"status": "updated"},
    )
    assert {r.id for r in results} == {chunk_a.id, chunk_b.id}

    # Remove items with tag==skip
    store.remove(
        "test_collection_filters",
        ids=[chunk_a.id, chunk_b.id, chunk_c.id],
        filters={"tag": "skip"},
    )

    remaining = store.search(
        collection_name="test_collection_filters",
        query_vector=[0.0, 0.0, 0.0, 0.0],
        k=10,
    )
    assert {r.id for r in remaining} == {chunk_a.id, chunk_b.id}


@pytest.mark.integration
def test_pgvector_batched_add_and_close(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres, use_pool=True)

    store.create_collection(
        "test_batched",
        vector_config=[VectorConfig(name="embedding", dimensions=2)],
    )

    # test batched insertion with executemany
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            text=f"batched {i}",
            embeddings=[DenseEmbedding(name="embedding", vector=[float(i), float(i)])],
        )
        for i in range(10)
    ]
    
    store.add(chunks, collection_name="test_batched")

    results = store.search(
        collection_name="test_batched",
        query_vector=[0.0, 0.0],
        k=20,
    )
    assert len(results) == 10

    # Ensure explicit close works properly without raising errors
    assert store._pool is not None
    store.close()
    assert store._pool is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pgvector_async_batched_add_and_aclose(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres, use_pool=True)

    # Sync create collection since a_create_collection is typically not required or missing in simple cases
    # Assuming create_collection is sync only in this test context
    store.create_collection(
        "test_async_batched",
        vector_config=[VectorConfig(name="embedding", dimensions=2)],
    )

    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            text=f"async batched {i}",
            embeddings=[DenseEmbedding(name="embedding", vector=[float(i), float(i)])],
        )
        for i in range(10)
    ]
    
    await store.a_add(chunks, collection_name="test_async_batched")

    # Assuming a_search or search works
    results = store.search(
        collection_name="test_async_batched",
        query_vector=[0.0, 0.0],
        k=20,
    )
    assert len(results) == 10

    assert store._a_pool is not None
    await store.aclose()
    assert store._a_pool is None
