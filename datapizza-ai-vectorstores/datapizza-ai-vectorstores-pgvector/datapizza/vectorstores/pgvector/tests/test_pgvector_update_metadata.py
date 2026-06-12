import uuid

import pytest
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import Chunk, DenseEmbedding

from datapizza.vectorstores.pgvector import PgVectorVectorstore


@pytest.mark.integration
def test_pgvector_update_metadata_merges(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    store.create_collection(
        "update_metadata",
        vector_config=[VectorConfig(name="embedding", dimensions=4)],
    )

    chunk_id = str(uuid.uuid4())
    store.add(
        Chunk(
            id=chunk_id,
            text="hello",
            embeddings=[DenseEmbedding(name="embedding", vector=[0.0, 0.0, 0.0, 0.0])],
            metadata={"foo": "bar"},
        ),
        collection_name="update_metadata",
    )

    # Merge new metadata keys without dropping existing ones
    store.update(
        collection_name="update_metadata",
        payload={"baz": "qux"},
        points=[chunk_id],
    )

    results = store.retrieve("update_metadata", ids=[chunk_id])
    assert len(results) == 1
    assert results[0].metadata["foo"] == "bar"
    assert results[0].metadata["baz"] == "qux"
