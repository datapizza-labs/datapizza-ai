import asyncio
import contextlib
import uuid

import pytest

from datapizza.core.vectorstore import VectorConfig
from datapizza.type import Chunk, DenseEmbedding, SparseEmbedding
from datapizza.vectorstores.redis import RedisVectorstore


def unique_collection(prefix="test"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def store():
    try:
        vector_store = RedisVectorstore()
        vector_store.get_client().ping()
    except Exception as e:
        pytest.skip(f"Cannot connect to Redis: {e}")

    smoke_collection = unique_collection("smoke")
    try:
        vector_store.create_collection(
            smoke_collection,
            [VectorConfig(name="dense_emb_name", dimensions=2)],
        )
    except Exception as e:
        pytest.skip(f"Redis vector store dependencies are not available: {e}")
    finally:
        with contextlib.suppress(Exception):
            vector_store.delete_collection(smoke_collection)

    return vector_store


@pytest.fixture
def collection(store):
    collection_name = unique_collection()
    store.create_collection(
        collection_name,
        [VectorConfig(name="dense_emb_name", dimensions=2)],
    )
    yield collection_name
    with contextlib.suppress(Exception):
        store.delete_collection(collection_name)


def run_async(store, coroutine):
    async def runner():
        try:
            return await coroutine
        finally:
            if hasattr(store, "a_client"):
                await store.a_client.aclose()
                delattr(store, "a_client")

    return asyncio.run(runner())


def test_redis_vectorstore_a_add_and_a_search(store, collection):
    chunk = Chunk(
        id=str(uuid.uuid4()),
        text="Hello world",
        embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.1, 0.2])],
        metadata={"source": "test"},
    )

    async def add_and_search():
        await store.a_add(chunk, collection_name=collection)
        return await store.a_search(
            collection_name=collection,
            query_vector=DenseEmbedding(name="dense_emb_name", vector=[0.1, 0.2]),
            k=1,
        )

    results = run_async(store, add_and_search())

    assert len(results) == 1
    assert results[0].id == chunk.id
    assert results[0].text == chunk.text
    assert results[0].metadata == chunk.metadata
    assert results[0].embeddings[0].name == "dense_emb_name"
    assert results[0].embeddings[0].vector == pytest.approx([0.1, 0.2], abs=1e-2)


def test_redis_vectorstore_a_add_raises_if_collection_missing(store):
    chunk = Chunk(
        id=str(uuid.uuid4()),
        text="Hello world",
        embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.1, 0.2])],
    )

    with pytest.raises(ValueError, match="Collection must be create to use it"):
        run_async(store, store.a_add(chunk, collection_name=unique_collection("missing")))


def test_redis_vectorstore_a_add_raises_on_sparse_embedding(store, collection):
    chunk = Chunk(
        id=str(uuid.uuid4()),
        text="Hello world",
        embeddings=[SparseEmbedding(name="sparse", values=[0.1], indices=[1])],
    )

    with pytest.raises(ValueError, match="support only dense embeddings"):
        run_async(store, store.a_add(chunk, collection_name=collection))
