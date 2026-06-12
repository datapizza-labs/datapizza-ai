import uuid

import pytest
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import Chunk, DenseEmbedding

from datapizza.vectorstores.pgvector import PgVectorVectorstore


@pytest.mark.integration
def test_pgvector_multi_vector_requires_vector_name(pgvector_postgres):
    store = PgVectorVectorstore(dsn=pgvector_postgres)

    # Create a collection with two vector fields.
    store.create_collection(
        "multi_vector",
        vector_config=[
            VectorConfig(name="openai", dimensions=4),
            VectorConfig(name="cohere", dimensions=4),
        ],
    )

    chunk_id = str(uuid.uuid4())
    store.add(
        Chunk(
            id=chunk_id,
            text="multi",
            embeddings=[
                DenseEmbedding(name="openai", vector=[0.0, 0.0, 0.0, 0.0]),
                DenseEmbedding(name="cohere", vector=[0.1, 0.1, 0.1, 0.1]),
            ],
        ),
        collection_name="multi_vector",
    )

    # Query must specify vector name in multi-vector collections.
    with pytest.raises(ValueError):
        store.search(collection_name="multi_vector", query_vector=[0.0, 0.0, 0.0, 0.0])

    # Works when specifying vector name
    results = store.search(
        collection_name="multi_vector",
        query_vector=[0.0, 0.0, 0.0, 0.0],
        vector_name="openai",
        k=1,
    )

    assert len(results) == 1
    assert results[0].text == "multi"
