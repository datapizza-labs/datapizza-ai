import pytest

from datapizza.vectorstores.pgvector import PgVectorVectorstore


def test_pgvector_vectorstore_imports():
    # Basic smoke test to validate the package is importable and the class is available.
    store = PgVectorVectorstore(dsn="postgresql://localhost:5432/postgres")
    assert store is not None


@pytest.mark.parametrize("schema", ["public", "test_schema"])
def test_pgvector_vectorstore_schema(schema):
    store = PgVectorVectorstore(dsn="postgresql://localhost:5432/postgres", schema=schema)
    assert store.schema == schema
