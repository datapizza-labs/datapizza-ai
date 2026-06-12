"""PostgreSQL + pgvector vectorstore implementation.

This implementation is intended to match the `Vectorstore` interface from
`datapizza-ai-core`, allowing users to store and retrieve embeddings using
Postgres + pgvector.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any, cast, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pgvector.psycopg import Vector, register_vector, register_vector_async

try:
    from psycopg_pool import AsyncConnectionPool, ConnectionPool
except ImportError:  # pragma: no cover
    ConnectionPool = None  # type: ignore[assignment]
    AsyncConnectionPool = None  # type: ignore[assignment]

from datapizza.core.vectorstore import Distance, VectorConfig, Vectorstore
from datapizza.type import Chunk, DenseEmbedding, EmbeddingFormat, SparseEmbedding

log = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid identifier '{name}'. Only alphanumeric characters and underscores are allowed, and it must not start with a digit."
        )
    return name


class PgVectorVectorstore(Vectorstore):
    """A vectorstore implementation backed by Postgres + pgvector."""

    def __init__(
        self,
        dsn: str | None = None,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        schema: str = "public",
        use_pool: bool = False,
        pool_min_size: int = 1,
        pool_max_size: int = 10,
        pool_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize the PgVectorVectorstore."""
        self.dsn = dsn
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.schema = schema
        self.use_pool = use_pool
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self.pool_kwargs = pool_kwargs or {}
        self.kwargs: dict[str, Any] = kwargs

        # Cached connections (if not using a pool)
        self._conn: Optional[psycopg.Connection] = None
        self._a_conn: Optional[psycopg.AsyncConnection] = None

        # Optional pools
        self._pool: Optional[ConnectionPool] = None
        self._a_pool: Optional[AsyncConnectionPool] = None

        # Local cache of collection vector configs (for name inference)
        self._collections: Dict[str, List[VectorConfig]] = {}

    def _qualified_table(self, collection_name: str) -> str:
        """Return a safely quoted schema-qualified table name."""
        schema = _validate_identifier(self.schema)
        """Return a safely quoted schema-qualified table name."""
        schema = _validate_identifier(self.schema)
        table = _validate_identifier(collection_name)
        return f'"{schema}"."{table}"'

    def _execute(self, cur: Any, query: Any, params: Any = None):
        """Execute on a cursor with relaxed typing for psycopg's execute signature."""
        if params is None:
            return cast(Any, cur).execute(query)
        return cast(Any, cur).execute(query, params)

    async def _a_execute(self, cur: Any, query: Any, params: Any = None):
        """Async execute wrapper with relaxed typing."""
        if params is None:
            return await cast(Any, cur).execute(query)
        return await cast(Any, cur).execute(query, params)

    def _normalize_dsn(self, dsn: str) -> str:
        """Normalize DSNs coming from SQLAlchemy/testcontainers."""
        if dsn.startswith("postgresql+psycopg2://"):
            return dsn.replace("postgresql+psycopg2://", "postgresql://", 1)
        return dsn

    def _build_dsn(self) -> str:
        """Construct a libpq-style DSN string from provided connection components."""
        if self.dsn:
            return self._normalize_dsn(self.dsn)

        parts: list[str] = []
        if self.host:
            parts.append(f"host={self.host}")
        if self.port:
            parts.append(f"port={self.port}")
        if self.database:
            parts.append(f"dbname={self.database}")
        if self.user:
            parts.append(f"user={self.user}")
        if self.password:
            parts.append(f"password={self.password}")
        return " ".join(parts)

    def _get_pool(self) -> ConnectionPool:
        if not self.use_pool:
            raise RuntimeError("Connection pooling is disabled")
        if ConnectionPool is None:
            raise ImportError(
                "psycopg_pool is required for connection pooling (pip install psycopg_pool)"
            )
        if self._pool is None:
            dsn = self._build_dsn()
            self._pool = ConnectionPool(
                dsn,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                open=True,
                **self.pool_kwargs,
            )
        return self._pool

    def _conn_context(self):
        """Return a context manager yielding a connection (pooled or single)."""
        if self.use_pool:
            pool = self._get_pool()

            @contextlib.contextmanager
            def _ctx():
                with pool.connection() as conn:
                    # Ensure pgvector is registered on each pooled connection.
                    register_vector(conn)
                    yield conn

            return _ctx()

        @contextlib.contextmanager
        def _ctx():
            conn = self._get_conn()
            yield conn

        return _ctx()

    async def _get_a_pool(self) -> AsyncConnectionPool:
        if not self.use_pool:
            raise RuntimeError("Connection pooling is disabled")
        if AsyncConnectionPool is None:
            raise ImportError(
                "psycopg_pool is required for async connection pooling (pip install psycopg_pool)"
            )
        if self._a_pool is None:
            dsn = self._build_dsn()
            # Avoid deprecated behavior: do not open the pool in the constructor.
            self._a_pool = AsyncConnectionPool(
                dsn,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                open=False,
                **self.pool_kwargs,
            )
            await self._a_pool.open()
        return self._a_pool

    async def _a_conn_context(self):
        """Return an async context manager yielding a connection."""
        if self.use_pool:
            pool = await self._get_a_pool()

            @contextlib.asynccontextmanager
            async def _ctx():
                async with pool.connection() as conn:
                    # Ensure pgvector is registered on each pooled connection.
                    await register_vector_async(conn)
                    yield conn

            return _ctx()

        @contextlib.asynccontextmanager
        async def _ctx():
            conn = await self._get_a_conn()
            yield conn

        return _ctx()

    def _build_filter_clause(
        self, filters: Optional[Dict[str, Any]] = None, prefix: str = "WHERE"
    ) -> tuple[str, list[Any]]:
        """Build an optional SQL clause for JSONB metadata filters.

        Args:
            filters: A dict of metadata fields to match via JSONB containment.
            prefix: SQL clause prefix, e.g. "WHERE" or "AND".
        """
        if not filters:
            return "", []

        # Use JSONB containment for simple metadata filtering
        return f"{prefix} metadata @> %s", [Jsonb(filters)]

    def _get_conn(self) -> psycopg.Connection:
        if self._conn is None:
            conn_args = self.kwargs.copy()

            if self.dsn:
                # psycopg expects a normal libpq connection string; testcontainers gives
                # SQLAlchemy-style strings like postgresql+psycopg2://...
                dsn = self.dsn
                if dsn.startswith("postgresql+psycopg2://"):
                    dsn = dsn.replace("postgresql+psycopg2://", "postgresql://", 1)
                conn = psycopg.connect(dsn, **conn_args)
            else:
                if self.host:
                    conn_args["host"] = self.host
                if self.port:
                    conn_args["port"] = self.port
                if self.database:
                    conn_args["dbname"] = self.database
                if self.user:
                    conn_args["user"] = self.user
                if self.password:
                    conn_args["password"] = self.password
                conn = psycopg.connect(**conn_args)

            register_vector(conn)
            conn.autocommit = True
            self._conn = conn
        return self._conn

    async def _get_a_conn(self) -> psycopg.AsyncConnection:
        if self._a_conn is None:
            conn_args = self.kwargs.copy()

            if self.dsn:
                dsn = self.dsn
                if dsn.startswith("postgresql+psycopg2://"):
                    dsn = dsn.replace("postgresql+psycopg2://", "postgresql://", 1)
                conn = await psycopg.AsyncConnection.connect(dsn, **conn_args)
            else:
                if self.host:
                    conn_args["host"] = self.host
                if self.port:
                    conn_args["port"] = self.port
                if self.database:
                    conn_args["dbname"] = self.database
                if self.user:
                    conn_args["user"] = self.user
                if self.password:
                    conn_args["password"] = self.password
                conn = await psycopg.AsyncConnection.connect(**conn_args)

            # Use the async registration helper for async connections
            await register_vector_async(conn)
            await conn.set_autocommit(True)
            self._a_conn = conn
        return self._a_conn

    def create_collection(
        self,
        collection_name: str,
        vector_config: list[VectorConfig],
        create_index: bool = False,
        index_method: str = "hnsw",
        index_name: Optional[str] = None,
        ivfflat_lists: Optional[int] = None,
        **kwargs,
    ):
        """Create a table for the given collection name and vector configuration.

        Args:
            collection_name: Name of the collection/table.
            vector_config: List of VectorConfig describing columns.
            create_index: If True, create an ANN index immediately.
            index_method: One of "hnsw" or "ivfflat".
            index_name: Optional explicit index name.
            ivfflat_lists: Required when using ivfflat.
        """
        if not collection_name:
            raise ValueError("collection_name must be provided")

        table = self._qualified_table(collection_name)

        # Build a normalized schema description for the provided config.
        requested_meta = [
            {
                "name": cfg.name,
                "dimensions": cfg.dimensions,
                "distance": cfg.distance.value if hasattr(cfg, "distance") else None,
                "format": cfg.format.value if hasattr(cfg, "format") else None,
            }
            for cfg in vector_config
        ]

        # If the collection was already created, ensure the requested schema matches.
        existing_meta = self._read_meta_config(collection_name)
        if existing_meta is not None:
            if existing_meta != requested_meta:
                raise ValueError(
                    f"Collection '{collection_name}' already exists with a different vector schema. "
                    "Call drop_collection() first if you want to recreate it."
                )
            # Cache the existing config for name inference.
            self._collections[collection_name] = [VectorConfig(**m) for m in existing_meta]
            return

        # Cache the requested config for name inference.
        self._collections[collection_name] = vector_config

        # Build schema
        cols = [
            "id TEXT PRIMARY KEY",
            "text TEXT NOT NULL",
            "metadata JSONB NOT NULL DEFAULT '{}'",
        ]

        for cfg in vector_config:
            if cfg.format != EmbeddingFormat.DENSE:
                log.warning(
                    "Skipping non-dense vector config %s: pgvector only supports dense embeddings.",
                    cfg,
                )
                continue
            name = cfg.name or "embedding"
            _validate_identifier(name)
            cols.append(f'"{name}" vector({cfg.dimensions})')

        with self._conn_context() as conn:
            with conn.cursor() as cur:
                self._execute(cur, f"CREATE SCHEMA IF NOT EXISTS \"{self.schema}\";")
                self._execute(cur, f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(cols)});")

        # Validate that the created table matches the requested config if possible.
        # This is best-effort: some pgvector / Postgres versions may not expose the
        # vector type metadata in a way we can reliably introspect.
        actual_config = self._load_collection_config_from_db(collection_name)
        if actual_config is None:
            log.warning(
                "Unable to introspect collection '%s' after creation; continuing without schema verification.",
                collection_name,
            )
        else:
            def _schema_key(cfg: VectorConfig) -> tuple[str | None, int | None]:
                return (cfg.name, cfg.dimensions)

            requested_keys = {_schema_key(c) for c in vector_config}
            actual_keys = {_schema_key(c) for c in actual_config}
            if requested_keys != actual_keys:
                raise ValueError(
                    f"Collection '{collection_name}' exists with a different schema: "
                    f"requested={requested_keys} actual={actual_keys}"
                )

        # Persist config to the metadata table for faster subsequent loads.
        self._write_meta_config(collection_name, requested_meta)

        if create_index:
            self.create_index(
                collection_name=collection_name,
                vector_name=None,
                method=index_method,
                index_name=index_name,
                ivfflat_lists=ivfflat_lists,
            )

    def _chunk_to_row(self, chunk: Chunk) -> Dict[str, Any]:
        if not chunk.embeddings:
            raise ValueError("Chunk must have an embedding")

        row: Dict[str, Any] = {
            "id": chunk.id,
            "text": chunk.text or "",
            "metadata": Jsonb(chunk.metadata or {}),
        }

        for emb in chunk.embeddings:
            if isinstance(emb, DenseEmbedding):
                name = emb.name or "embedding"
                row[name] = Vector(emb.vector)
            elif isinstance(emb, SparseEmbedding):
                raise ValueError("Sparse embeddings are not supported by pgvector")
            else:
                raise ValueError(f"Unsupported embedding type: {type(emb)}")

        return row

    def add(self, chunk: Chunk | list[Chunk], collection_name: str | None = None):
        if not collection_name:
            raise ValueError("collection_name must be provided")

        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        if not chunks:
            return

        rows = [self._chunk_to_row(c) for c in chunks]
        first_row = rows[0]

        table = self._qualified_table(collection_name)
        # Assuming all chunks uniformly share the same metadata struct & embedding names config 
        cols = ", ".join(first_row.keys())
        placeholders = ", ".join(["%s"] * len(first_row))
        updates = ", ".join([f"{k}=EXCLUDED.{k}" for k in first_row.keys() if k != "id"])
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) ON CONFLICT (id) DO UPDATE SET {updates};"
        
        params_seq = [list(row.values()) for row in rows]

        with self._conn_context() as conn:
            with conn.cursor() as cur:
                # Use psycopg's executemany for batch inserts
                cur.executemany(sql, params_seq)

    async def a_add(
        self, chunk: Chunk | list[Chunk], collection_name: str | None = None
    ):
        if not collection_name:
            raise ValueError("collection_name must be provided")

        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        if not chunks:
            return

        rows = [self._chunk_to_row(c) for c in chunks]
        first_row = rows[0]

        table = self._qualified_table(collection_name)
        cols = ", ".join(first_row.keys())
        placeholders = ", ".join(["%s"] * len(first_row))
        updates = ", ".join([f"{k}=EXCLUDED.{k}" for k in first_row.keys() if k != "id"])
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) ON CONFLICT (id) DO UPDATE SET {updates};"
        
        params_seq = [list(row.values()) for row in rows]

        async with await self._a_conn_context() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(sql, params_seq)

    def _meta_table_name(self) -> str:
        return f'"{self.schema}"."_datapizza_vectorstore_meta"'

    def _ensure_meta_table(self) -> None:
        """Ensure the metadata table exists for storing collection configs."""
        with self._conn_context() as conn:
            with conn.cursor() as cur:
                self._execute(
                    cur,
                    f"CREATE TABLE IF NOT EXISTS {self._meta_table_name()} ("
                    "collection_name TEXT PRIMARY KEY, "
                    "config JSONB NOT NULL"
                    ");",
                )

    def _write_meta_config(self, collection_name: str, config: list[Dict[str, Any]]) -> None:
        self._ensure_meta_table()
        with self._conn_context() as conn:
            with conn.cursor() as cur:
                self._execute(
                    cur,
                    f"INSERT INTO {self._meta_table_name()} (collection_name, config) VALUES (%s, %s) "
                    "ON CONFLICT (collection_name) DO UPDATE SET config = EXCLUDED.config;",
                    (collection_name, Jsonb(config)),
                )

    def _read_meta_config(self, collection_name: str) -> Optional[list[Dict[str, Any]]]:
        try:
            with self._conn_context() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    self._execute(
                        cur,
                        f"SELECT config FROM {self._meta_table_name()} WHERE collection_name = %s;",
                        (collection_name,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    return row["config"]
        except Exception:
            return None

    def _load_collection_config_from_db(
        self, collection_name: str
    ) -> Optional[list[VectorConfig]]:
        """Rebuild _collections cache by introspecting the Postgres schema.

        Uses a lightweight metadata table when available, falling back to
        information_schema introspection.
        """
        # Prefer stored config when available (fast + reliable)
        meta = self._read_meta_config(collection_name)
        if meta:
            try:
                configs = [VectorConfig(**c) for c in meta]
                self._collections[collection_name] = configs
                return configs
            except Exception:
                pass

        # a.atttypmod stores the vector dimension for pgvector (as 4 + 4*dim)
        sql = (
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod) as typ, a.atttypmod "
            "FROM pg_attribute a "
            "JOIN pg_class c ON a.attrelid = c.oid "
            "JOIN pg_namespace n ON c.relnamespace = n.oid "
            "WHERE n.nspname = %s AND c.relname = %s "
            "  AND a.attnum > 0 AND NOT a.attisdropped "
            "  AND format_type(a.atttypid, a.atttypmod) LIKE '%vector%';"
        )
        schema = self.schema
        table_name = collection_name

        try:
            with self._conn_context() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    self._execute(cur, sql, (schema, table_name))
                    rows = cur.fetchall()
        except Exception:
            # If we can't query (or the table doesn't exist), treat as missing.
            return None

        vector_configs: list[VectorConfig] = []
        for row in rows:
            typ = row.get("typ")
            atttypmod = row.get("atttypmod")

            dim: Optional[int] = None

            # typ can be 'vector' or 'vector(N)'
            if isinstance(typ, str) and typ.startswith("vector(") and typ.endswith(")"):
                try:
                    dim = int(typ[len("vector(") : -1])
                except ValueError:
                    dim = None
            elif isinstance(typ, str) and typ == "vector" and isinstance(atttypmod, int):
                # pgvector uses atttypmod = 4 + 4*dim
                try:
                    dim = (atttypmod - 4) // 4
                except Exception:
                    dim = None

            if dim is None:
                continue

            vector_configs.append(
                VectorConfig(name=row["attname"], dimensions=dim)
            )

        if vector_configs:
            self._collections[collection_name] = vector_configs
            return vector_configs
        return None

    def list_collections(self) -> list[str]:
        """Return the list of collections (tables) tracked by this vectorstore."""
        try:
            with self._conn_context() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    self._execute(
                        cur,
                        f"SELECT collection_name FROM {self._meta_table_name()} ORDER BY collection_name;",
                    )
                    rows = cur.fetchall()
                    return [row["collection_name"] for row in rows]
        except Exception:
            # If the meta table does not exist, fall back to introspection.
            return []

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        """Return metadata about a collection (vector columns and dimensions)."""
        configs = self._load_collection_config_from_db(collection_name)
        if not configs:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        return {
            "name": collection_name,
            "schema": self.schema,
            "vectors": [
                {"name": cfg.name or "embedding", "dimensions": cfg.dimensions}
                for cfg in configs
            ],
        }

    def _get_vector_config(
        self, collection_name: str, vector_name: Optional[str]
    ) -> VectorConfig:
        configs = self._collections.get(collection_name)
        if not configs:
            configs = self._load_collection_config_from_db(collection_name)
        if not configs:
            raise ValueError(
                f"Collection '{collection_name}' is not registered or does not exist. "
                "Call create_collection first."
            )

        dense_configs = [c for c in configs if c.format == EmbeddingFormat.DENSE]
        if not dense_configs:
            raise ValueError(
                f"Collection '{collection_name}' has no dense vectors configured"
            )

        if vector_name:
            for cfg in dense_configs:
                if (cfg.name or "embedding") == vector_name:
                    return cfg
            raise ValueError(
                f"Vector name '{vector_name}' not found in collection '{collection_name}'"
            )

        if len(dense_configs) == 1:
            return dense_configs[0]

        raise ValueError(
            f"Collection '{collection_name}' has multiple vector columns; specify `vector_name`."
        )

    def _get_vector_column(self, collection_name: str, vector_name: Optional[str]) -> str:
        cfg = self._get_vector_config(collection_name, vector_name)
        return cfg.name or "embedding"

    def _row_to_chunk(self, row: dict, vector_name: Optional[str]) -> Chunk:
        emb: List[Any] = []
        if vector_name and vector_name in row:
            val = row[vector_name]
            if val is not None:
                emb.append(DenseEmbedding(name=vector_name, vector=list(val)))

        return Chunk(
            id=str(row["id"]),
            text=row.get("text", ""),
            embeddings=emb,
            metadata=row.get("metadata", {}) or {},
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        vector_name: str | None = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> list[Chunk]:
        if not collection_name:
            raise ValueError("collection_name must be provided")

        cfg = self._get_vector_config(collection_name, vector_name)
        col = cfg.name or "embedding"
        table = self._qualified_table(collection_name)

        op = "<->" if cfg.distance == Distance.EUCLIDEAN else "<=>"

        filter_sql, filter_params = self._build_filter_clause(filters)
        sql = (
            f"SELECT id, text, metadata, {col} FROM {table} "
            f"{filter_sql} ORDER BY {col} {op} %s LIMIT %s;"
        )

        with self._conn_context() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                self._execute(cur, sql, [*filter_params, Vector(query_vector), k])
                rows = cur.fetchall()

        return [self._row_to_chunk(row, col) for row in rows]

    async def a_search(
        self,
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        vector_name: str | None = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> list[Chunk]:
        if not collection_name:
            raise ValueError("collection_name must be provided")

        col = self._get_vector_column(collection_name, vector_name)
        table = self._qualified_table(collection_name)

        cfg = self._get_vector_config(collection_name, vector_name)
        op = "<->" if cfg.distance == Distance.EUCLIDEAN else "<=>"

        filter_sql, filter_params = self._build_filter_clause(filters)
        sql = (
            f"SELECT id, text, metadata, {col} FROM {table} "
            f"{filter_sql} ORDER BY {col} {op} %s LIMIT %s;"
        )

        async with await self._a_conn_context() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await self._a_execute(
                    cur, sql, [*filter_params, Vector(query_vector), k]
                )
                rows = await cur.fetchall()

        return [self._row_to_chunk(row, col) for row in rows]

    def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> list[Chunk]:
        if not collection_name:
            raise ValueError("collection_name must be provided")
        if not ids:
            return []

        table = self._qualified_table(collection_name)
        filter_sql, filter_params = self._build_filter_clause(filters, prefix="AND")
        sql = f"SELECT id, text, metadata FROM {table} WHERE id = ANY(%s) {filter_sql};"

        with self._conn_context() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                self._execute(cur, sql, [ids, *filter_params])
                rows = cur.fetchall()

        return [self._row_to_chunk(row, None) for row in rows]

    def remove(
        self,
        collection_name: str,
        ids: list[str],
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if not collection_name:
            raise ValueError("collection_name must be provided")
        if not ids:
            return

        table = self._qualified_table(collection_name)
        filter_sql, filter_params = self._build_filter_clause(filters, prefix="AND")
        sql = f"DELETE FROM {table} WHERE id = ANY(%s) {filter_sql};"

        with self._conn_context() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, [ids, *filter_params])

    def update(
        self,
        collection_name: str,
        payload: dict,
        points: list[str | int],
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Update metadata for a set of points.

        Args:
            collection_name: Target collection.
            payload: Metadata changes to merge (JSONB merge semantics).
            points: List of IDs to update.
        """
        if not collection_name:
            raise ValueError("collection_name must be provided")
        if not points:
            return

        table = self._qualified_table(collection_name)
        filter_sql, filter_params = self._build_filter_clause(filters, prefix="AND")
        sql = (
            f"UPDATE {table} SET metadata = metadata || %s WHERE id = ANY(%s) "
            f"{filter_sql};"
        )
        with self._conn_context() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, [Jsonb(payload), points, *filter_params])

    def _get_ops_class(self, distance: Distance) -> str:
        return "vector_l2_ops" if distance == Distance.EUCLIDEAN else "vector_cosine_ops"

    def close(self):
        """Close any open synchronous or asynchronous connections and pools."""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
        if self._pool is not None:
            try:
                self._pool.close()
            finally:
                self._pool = None

    async def aclose(self):
        """Close any open asynchronous connection (and sync connection) and pools."""
        if self._a_conn is not None:
            try:
                await self._a_conn.close()
            finally:
                self._a_conn = None
        if self._a_pool is not None:
            try:
                await self._a_pool.close()
            finally:
                self._a_pool = None

        # Close sync connection too, for safety.
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
        if self._pool is not None:
            try:
                self._pool.close()
            finally:
                self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    def drop_collection(self, collection_name: str, if_exists: bool = True):
        """Drop the collection table from the database."""
        if not collection_name:
            raise ValueError("collection_name must be provided")

        table = self._qualified_table(collection_name)
        with self._conn_context() as conn:
            with conn.cursor() as cur:
                if if_exists:
                    self._execute(cur, f"DROP TABLE IF EXISTS {table};")
                else:
                    self._execute(cur, f"DROP TABLE {table};")

        self._collections.pop(collection_name, None)

    async def a_drop_collection(self, collection_name: str, if_exists: bool = True):
        """Async drop of the collection table."""
        if not collection_name:
            raise ValueError("collection_name must be provided")

        table = self._qualified_table(collection_name)
        async with await self._a_conn_context() as conn:
            async with conn.cursor() as cur:
                if if_exists:
                    await self._a_execute(cur, f"DROP TABLE IF EXISTS {table};")
                else:
                    await self._a_execute(cur, f"DROP TABLE {table};")

        self._collections.pop(collection_name, None)

    def create_index(
        self,
        collection_name: str,
        vector_name: Optional[str] = None,
        method: str = "hnsw",
        index_name: Optional[str] = None,
        ivfflat_lists: Optional[int] = None,
    ):
        """Create an ANN index for the given vector column.

        Args:
            collection_name: collection/table name.
            vector_name: vector column name (required for multi-vector collections).
            method: One of "hnsw" or "ivfflat".
            index_name: Optional explicit index name; generated if missing.
            ivfflat_lists: Required for ivfflat; ignored for hnsw.
        """
        if not collection_name:
            raise ValueError("collection_name must be provided")

        cfg = self._get_vector_config(collection_name, vector_name)
        col = cfg.name or "embedding"
        table = self._qualified_table(collection_name)

        ops = self._get_ops_class(cfg.distance)
        method = method.lower()
        if method not in {"hnsw", "ivfflat"}:
            raise ValueError("method must be 'hnsw' or 'ivfflat'")

        if not index_name:
            index_name = f"{collection_name}_{col}_{method}_idx"

        if method == "hnsw":
            sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} USING hnsw ({col} {ops});"
        else:
            if ivfflat_lists is None:
                raise ValueError("ivfflat requires ivfflat_lists to be provided")
            sql = (
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} "
                f"USING ivfflat ({col} {ops}) WITH (lists = {ivfflat_lists});"
            )

        with self._conn_context() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql)

    async def a_create_index(
        self,
        collection_name: str,
        vector_name: Optional[str] = None,
        method: str = "hnsw",
        index_name: Optional[str] = None,
        ivfflat_lists: Optional[int] = None,
    ):
        """Async version of create_index."""
        if not collection_name:
            raise ValueError("collection_name must be provided")

        cfg = self._get_vector_config(collection_name, vector_name)
        col = cfg.name or "embedding"
        table = self._qualified_table(collection_name)

        ops = self._get_ops_class(cfg.distance)
        method = method.lower()
        if method not in {"hnsw", "ivfflat"}:
            raise ValueError("method must be 'hnsw' or 'ivfflat'")

        if not index_name:
            index_name = f"{collection_name}_{col}_{method}_idx"

        if method == "hnsw":
            sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} USING hnsw ({col} {ops});"
        else:
            if ivfflat_lists is None:
                raise ValueError("ivfflat requires ivfflat_lists to be provided")
            sql = (
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} "
                f"USING ivfflat ({col} {ops}) WITH (lists = {ivfflat_lists});"
            )

        async with await self._a_conn_context() as conn:
            async with conn.cursor() as cur:
                await self._a_execute(cur, sql)

    def list_indexes(self, collection_name: str) -> list[dict[str, Any]]:
        """List known indexes for a collection."""
        if not collection_name:
            raise ValueError("collection_name must be provided")

        sql = (
            "SELECT indexname, indexdef "
            "FROM pg_indexes "
            "WHERE schemaname = %s AND tablename = %s;"
        )
        with self._conn_context() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                self._execute(cur, sql, (self.schema, collection_name))
                rows = cur.fetchall()

        return [
            {"name": r["indexname"], "definition": r["indexdef"]} for r in rows
        ]

    def describe_index(self, index_name: str) -> dict[str, Any]:
        """Return information about a specific index."""
        if not index_name:
            raise ValueError("index_name must be provided")

        sql = (
            "SELECT schemaname, tablename, indexname, indexdef "
            "FROM pg_indexes "
            "WHERE schemaname = %s AND indexname = %s;"
        )
        with self._conn_context() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                self._execute(cur, sql, (self.schema, index_name))
                row = cur.fetchone()

        if not row:
            raise ValueError(f"Index '{index_name}' does not exist")

        definition = row["indexdef"]

        method = None
        ivfflat_lists = None
        if definition:
            if "using hnsw" in definition.lower():
                method = "hnsw"
            elif "using ivfflat" in definition.lower():
                method = "ivfflat"
                # parse lists=...
                import re

                m = re.search(r"lists\s*=\s*'?(\d+)'?", definition, re.IGNORECASE)
                if m:
                    ivfflat_lists = int(m.group(1))

        return {
            "schema": row["schemaname"],
            "table": row["tablename"],
            "name": row["indexname"],
            "definition": definition,
            "method": method,
            "ivfflat_lists": ivfflat_lists,
        }

    def drop_index(self, index_name: str, if_exists: bool = True):
        """Drop an index by name."""
        if not index_name:
            raise ValueError("index_name must be provided")

        stmt = f"DROP INDEX {'IF EXISTS ' if if_exists else ''}{self.schema}.{index_name};"
        with self._conn_context() as conn:
            with conn.cursor() as cur:
                self._execute(cur, stmt)

    async def a_drop_index(self, index_name: str, if_exists: bool = True):
        """Async drop of an index."""
        if not index_name:
            raise ValueError("index_name must be provided")

        stmt = f"DROP INDEX {'IF EXISTS ' if if_exists else ''}{self.schema}.{index_name};"
        async with await self._a_conn_context() as conn:
            async with conn.cursor() as cur:
                await self._a_execute(cur, stmt)
