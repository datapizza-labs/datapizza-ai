import pytest


@pytest.fixture(scope="session")
def pgvector_postgres():
    """Start a Postgres container with pgvector enabled."""
    try:
        import psycopg
        from testcontainers.postgres import PostgresContainer

        with PostgresContainer("pgvector/pgvector:pg16") as pg:
            dsn = pg.get_connection_url()
            # Normalize SQLAlchemy style DSN to libpq format (psycopg expects it).
            if dsn.startswith("postgresql+psycopg2://"):
                dsn = dsn.replace("postgresql+psycopg2://", "postgresql://", 1)

            # Ensure pgvector extension is installed in the database.
            with psycopg.connect(dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            yield dsn
    except Exception as e:
        pytest.skip(
            f"Skipping integration tests because docker/testcontainers is unavailable: {e}"
        )
