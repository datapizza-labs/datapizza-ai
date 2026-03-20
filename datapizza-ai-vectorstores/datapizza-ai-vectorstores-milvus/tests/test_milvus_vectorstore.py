import contextlib
import uuid

import pytest
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import (
    Chunk,
    EmbeddingFormat,
    SparseEmbedding,
)

from datapizza.vectorstores.milvus import MilvusVectorstore

MILVUS_URI = "./milvus.db"


def unique_coll(prefix="itest"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def make_chunk(cid: str, text: str = "hello", meta: bool = True) -> Chunk:
    metadata = {"source": "itest", "idx": cid} if meta else None
    return Chunk(
        id=cid,
        text=text,
        embeddings=[
            SparseEmbedding(
                name="sparse_vec",
                indices=[1, 5, 9],
                values=[0.5, 0.2, 0.1],
            )
        ],
        metadata=metadata,
    )


@pytest.fixture(scope="session")
def store():
    s = MilvusVectorstore(uri=MILVUS_URI)
    try:
        client = s.get_client()
        client.list_collections()
    except Exception as e:
        pytest.skip(f"Cannot connect to Milvus at {MILVUS_URI}: {e}")
    return s


@pytest.fixture
def vector_cfg():
    return [
        VectorConfig(
            name="sparse_vec",
            format=EmbeddingFormat.SPARSE,
            dimensions=1536,
            distance="Cosine",
        )
    ]


@pytest.fixture
def collection(store, vector_cfg):
    name = unique_coll()
    store.create_collection(
        collection_name=name,
        vector_config=vector_cfg,
    )
    yield name
    # Cleanup
    with contextlib.suppress(Exception):
        store.delete_collection(name)


def test_metric_from_config_sparse_cosine(store, vector_cfg):
    assert store._metric_from_config(vector_cfg[0]) == "COSINE"


def test_create_and_list_collections(store, vector_cfg):
    name = unique_coll()
    try:
        store.create_collection(name, vector_cfg)
        cols = store.get_collections()
        assert name in cols
    finally:
        store.delete_collection(name)


def test_add_and_retrieve_roundtrip(store, collection):
    c = make_chunk("id-1", text="alpha")
    store.add(c, collection_name=collection)
    # flush collection
    client = store.get_client()
    client.flush(collection_name=collection)
    out = store.retrieve(collection, ids=["id-1"])
    assert len(out) == 1
    got = out[0]
    assert got.id == "id-1"
    assert got.text == "alpha"
    assert got.metadata.get("source") == "itest"
    emb_names = {e.name for e in got.embeddings}
    assert "sparse_vec" in emb_names


def test_batch_add(store, collection):
    batch = [make_chunk(f"id-{i}") for i in range(6)]
    store.add(batch, collection_name=collection, batch_size=2)
    # flush collection
    client = store.get_client()
    client.flush(collection_name=collection)
    out = store.retrieve(collection, ids=[f"id-{i}" for i in range(6)])
    assert len(out) == 6


def test_update_upsert(store, collection):
    store.add(make_chunk("u1", text="v1"), collection_name=collection)
    store.update(collection, make_chunk("u1", text="v2"))
    got = store.retrieve(collection, ids=["u1"])[0]
    assert got.text == "v2"


def test_remove(store, collection):
    store.add(make_chunk("del-1"), collection_name=collection)
    store.remove(collection, ids=["del-1"])
    # flush collection
    client = store.get_client()
    client.flush(collection_name=collection)
    out = store.retrieve(collection, ids=["del-1"])
    assert out == []


def test_search_single_sparse(store, collection):
    store.add([make_chunk(f"s{i}") for i in range(5)], collection_name=collection)
    # flush collection
    client = store.get_client()
    client.flush(collection_name=collection)
    q = SparseEmbedding(name="sparse_vec", indices=[1, 2], values=[0.3, 0.7])
    res = store.search(collection, q, k=3)
    assert isinstance(res, list)
    assert len(res) == 3
    from datapizza.type import Chunk as _C

    assert all(isinstance(c, _C) for c in res)


def test_dump_collection_paged(store, collection):
    for i in range(7):
        store.add(make_chunk(f"d{i}"), collection_name=collection)
    # flush collection
    client = store.get_client()
    client.flush(collection_name=collection)
    dumped = list(store.dump_collection(collection, page_size=3))
    assert len(dumped) >= 7
    ids = {c.id for c in dumped}
    assert "d0" in ids and "d6" in ids
