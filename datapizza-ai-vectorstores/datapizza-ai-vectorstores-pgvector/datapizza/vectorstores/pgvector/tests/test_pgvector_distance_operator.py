import pytest

from datapizza.core.vectorstore import Distance, VectorConfig
from datapizza.type import Chunk, DenseEmbedding
from datapizza.vectorstores.pgvector import PgVectorVectorstore


class _FakeCursor:
    def __init__(self):
        self.queries = []
        self.params = []

    def execute(self, sql, params=None):
        self.queries.append(sql)
        self.params.append(params)

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _FakeConn:
    def __init__(self):
        self.cursor_obj = _FakeCursor()

    def cursor(self, *args, **kwargs):
        return self.cursor_obj


@pytest.mark.parametrize(
    "distance,expected_op",
    [
        (Distance.COSINE, "<=>"),
        (Distance.EUCLIDEAN, "<->"),
    ],
)
def test_pgvector_search_operator(distance, expected_op):
    """Verify that Distance maps to the correct pgvector operator."""

    store = PgVectorVectorstore(dsn="postgresql://user:pass@localhost:5432/db")
    store._collections["test"] = [VectorConfig(name="embedding", dimensions=4, distance=distance)]

    fake_conn = _FakeConn()
    store._get_conn = lambda: fake_conn

    store.search(
        collection_name="test",
        query_vector=[0.0, 0.0, 0.0, 0.0],
        k=1,
        vector_name="embedding",
    )

    assert len(fake_conn.cursor_obj.queries) == 1
    assert expected_op in fake_conn.cursor_obj.queries[0]
