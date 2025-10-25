import uuid

import pytest
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import EmbeddingFormat
from datapizza.type.type import Chunk, DenseEmbedding, SparseEmbedding

from datapizza.vectorstores.qdrant import QdrantVectorstore


@pytest.fixture
def vectorstore() -> QdrantVectorstore:
    vectorstore = QdrantVectorstore(location=":memory:")
    vectorstore.create_collection(
        collection_name="test",
        vector_config=[VectorConfig(dimensions=1536, name="dense_emb_name")],
    )

    return vectorstore


def test_qdrant_vectorstore_init():
    vectorstore = QdrantVectorstore(location=":memory:")
    assert vectorstore is not None


def test_qdrant_vectorstore_add(vectorstore):
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            text="Hello world",
            embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.0] * 1536)],
        )
    ]
    vectorstore.add(chunks, collection_name="test")

    res = vectorstore.search(collection_name="test", query_vector=[0.0] * 1536)
    assert len(res) == 1

    res = vectorstore.search(
        collection_name="test", query_vector=[0.0] * 1536, vector_name="dense_emb_name"
    )
    assert len(res) == 1


def test_qdrant_vectorstore_create_collection(vectorstore):
    vectorstore.create_collection(
        collection_name="test2",
        vector_config=[VectorConfig(dimensions=1536, name="test2")],
    )

    colls = vectorstore.get_collections()

    assert len(colls.collections) == 2


def test_delete_collection(vectorstore):
    vectorstore.create_collection(
        collection_name="deleteme",
        vector_config=[VectorConfig(dimensions=1536, name="test2")],
    )

    colls = vectorstore.get_collections()
    assert len(colls.collections) == 2
    vectorstore.delete_collection(collection_name="deleteme")

    colls = vectorstore.get_collections()
    assert len(colls.collections) == 1


def test_qdrant_create_collection_with_sparse_vector(vectorstore):
    vectorstore.create_collection(
        collection_name="test3",
        vector_config=[
            VectorConfig(dimensions=1536, name="test3", format=EmbeddingFormat.SPARSE)
        ],
    )

    dense = vectorstore.get_client().get_collection("test3").config.params.vectors
    sparse = (
        vectorstore.get_client().get_collection("test3").config.params.sparse_vectors
    )
    assert sparse is not None
    assert len(sparse) == 1
    assert len(dense) == 0


def test_qdrant_search_sparse_vector(vectorstore):
    vectorstore.create_collection(
        collection_name="sparse_test",
        vector_config=[VectorConfig(name="sparse", format=EmbeddingFormat.SPARSE)],
    )

    vectorstore.add(
        chunk=[
            Chunk(
                id=str(uuid.uuid4()),
                text="Hello world",
                embeddings=[SparseEmbedding(name="sparse", values=[0.1], indices=[1])],
            )
        ],
        collection_name="sparse_test",
    )

    results = vectorstore.search(
        collection_name="sparse_test",
        query_vector=SparseEmbedding(name="sparse", values=[0.1], indices=[1]),
    )
    assert len(results) == 1


def test_collection_with_multiple_vectors(vectorstore):
    vectorstore.create_collection(
        collection_name="multi_vector_test",
        vector_config=[
            VectorConfig(dimensions=1536, name="dense_emb_name"),
            VectorConfig(name="sparse", format=EmbeddingFormat.SPARSE),
        ],
    )

    vectorstore.add(
        chunk=[
            Chunk(
                id=str(uuid.uuid4()),
                text="Hello world",
                embeddings=[
                    DenseEmbedding(name="dense_emb_name", vector=[0.0] * 1536),
                    SparseEmbedding(name="sparse", values=[0.1], indices=[1]),
                ],
            )
        ],
        collection_name="multi_vector_test",
    )

    res_no_name = vectorstore.search(
        collection_name="multi_vector_test", query_vector=[0.0] * 1536
    )
    assert len(res_no_name) == 1

    res_dense_name = vectorstore.search(
        collection_name="multi_vector_test",
        query_vector=[0.0] * 1536,
        vector_name="dense_emb_name",
    )
    assert len(res_dense_name) == 1

    res_sparse_name = vectorstore.search(
        collection_name="multi_vector_test",
        query_vector=SparseEmbedding(name="sparse", values=[0.1], indices=[1]),
        vector_name="sparse",
    )
    assert len(res_sparse_name) == 1

    res_sparse_no_name = vectorstore.search(
        collection_name="multi_vector_test",
        query_vector=SparseEmbedding(name="sparse", values=[0.1], indices=[1]),
    )
    assert len(res_sparse_no_name) == 1
