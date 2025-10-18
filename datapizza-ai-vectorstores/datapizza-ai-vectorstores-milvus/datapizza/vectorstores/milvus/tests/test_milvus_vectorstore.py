import uuid
import pytest
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import Chunk, DenseEmbedding
from vector_store.milvus_vectorstore import MilvusVectorstore

M_HOST = "127.0.0.1"
M_PORT = "19530"
ALIAS = "default"


@pytest.fixture
def vectorstore() -> MilvusVectorstore:
    store = MilvusVectorstore(host=M_HOST, port=M_PORT, alias=ALIAS)
    store._connect()

    if "test" in store.get_collections():
        store.delete_collection("test")

    store.create_collection(
        collection_name="test",
        vector_config=[VectorConfig(dimensions=4, name="vector")],
    )

    store.create_index("test")
    store.load_collection("test")

    return store


def test_milvus_vectorstore_init(vectorstore):
    assert vectorstore is not None


def test_milvus_vectorstore_add(vectorstore):
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            text="Hello Milvus",
            embeddings=[DenseEmbedding(name="vector", vector=[0.0] * 4)],
        )
    ]
    vectorstore.add(chunks, collection_name="test")
    results = vectorstore.search("test", [0.0] * 4)
    assert len(results) == 1
    assert results[0].text == "Hello Milvus"


def test_milvus_vectorstore_create_collection(vectorstore):
    vectorstore.create_collection(
        collection_name="test2",
        vector_config=[VectorConfig(dimensions=4, name="vector")],
    )

    colls = vectorstore.get_collections()
    assert "test" in colls
    assert "test2" in colls


def test_delete_collection(vectorstore):
    vectorstore.create_collection(
        collection_name="deleteme",
        vector_config=[VectorConfig(dimensions=4, name="vector")],
    )
    colls = vectorstore.get_collections()
    assert "deleteme" in colls

    vectorstore.delete_collection("deleteme")
    colls = vectorstore.get_collections()
    assert "deleteme" not in colls
