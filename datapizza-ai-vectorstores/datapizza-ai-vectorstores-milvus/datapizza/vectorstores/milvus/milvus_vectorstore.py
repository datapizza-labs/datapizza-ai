import logging
from collections.abc import Generator
from typing import Any

from datapizza.core.vectorstore import VectorConfig, Vectorstore
from datapizza.type import (
    Chunk,
    DenseEmbedding,
    Embedding,
    EmbeddingFormat,
    SparseEmbedding,
)
from pymilvus import (
    AsyncMilvusClient,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    AsyncMilvusClient,
    MilvusException,
)
from pymilvus.milvus_client.index import IndexParams

log = logging.getLogger(__name__)


class MilvusVectorstore(Vectorstore):
    """
    Milvus Vectorstore with support for dense/sparse embeddings and hybrid search.
    """

    def __init__(
            self,
            uri: Optional[str] = None,
            host: Optional[str] = None,
            port: Optional[int] = None,
            user: Optional[str] = None,
            password: Optional[str] = None,
            secure: Optional[bool] = None,
            batch_size: int = 100,
            **connection_args: Any,
    ):
        """
        Initialize Milvus connection.

        Args:
            uri: Connection URI (e.g., "http://localhost:19530" or "./milvus.db" for Milvus Lite)
            host: Milvus server host
            port: Milvus server port
            user: Authentication username or empty
            password: Authentication password or empty
            secure: Use secure connection or empty
            batch_size: Default batch size for bulk operations
            **connection_args: Additional connection parameters
        """
        self.conn_kwargs: dict[str, Any] = {}
        if uri:
            self.conn_kwargs["uri"] = uri
        if host:
            self.conn_kwargs["host"] = host
        if port:
            self.conn_kwargs["port"] = port
        if user:
            self.conn_kwargs["user"] = user
        if password:
            self.conn_kwargs["password"] = password
        if secure is not None:
            self.conn_kwargs["secure"] = secure

        # Allow extra MilvusClient kwargs (e.g. token for Zilliz)
        self.conn_kwargs.update(connection_args or {})
        self.client: MilvusClient
        self.a_client: AsyncMilvusClient
        self.batch_size: int = 100

    def get_client(self) -> MilvusClient:
        if not hasattr(self, "client"):
            self._init_client()
        return self.client

    def _get_a_client(self) -> AsyncMilvusClient:
        if not hasattr(self, "a_client"):
            self._init_a_client()
        return self.a_client

    def _init_client(self):
        """Initialize synchronous client."""
        self.client = MilvusClient(**self.conn_kwargs)

    def _init_a_client(self):
        """Initialize asynchronous client."""
        self.a_client = AsyncMilvusClient(**self.conn_kwargs)

    @staticmethod
    def _chunk_to_row(chunk: Chunk) -> dict[str, Any]:
        """Convert Chunk to Milvus row format."""

        def _sparse_to_dict(se: SparseEmbedding) -> dict[int, float]:
            return {int(i): float(v) for i, v in zip(se.indices, se.values)}

        if not chunk.embeddings:
            raise ValueError("Chunk must have at least one embedding")

        row: dict[str, Any] = {
            "id": chunk.id,
            "text": chunk.text or "",
        }

        # add metadata
        if chunk.metadata:
            row.update(chunk.metadata)

        # add embeddings
        for e in chunk.embeddings:
            if isinstance(e, DenseEmbedding):
                row[e.name] = e.vector
            elif isinstance(e, SparseEmbedding):
                row[e.name] = _sparse_to_dict(e)
            else:
                raise ValueError(f"Unsupported embedding type: {type(e)}")

        return row

    @staticmethod
    def _entity_to_chunk(entity: dict[str, Any]) -> Chunk:
        """Convert Milvus entity to Chunk."""
        embeddings: list[Embedding] = []
        metadata: dict[str, Any] = {}

        for key, val in entity.items():
            if key in {"id", "text"}:
                continue

            # Dense vector
            if isinstance(val, list) and val and isinstance(val[0], (int, float)):
                embeddings.append(DenseEmbedding(name=key, vector=[float(x) for x in val]))
                continue

            # Sparse vector
            if isinstance(val, dict) and val:
                try:
                    items = sorted(((int(k), float(v)) for k, v in val.items()), key=lambda t: t[0])
                    indices = [i for i, _ in items]
                    values = [v for _, v in items]
                    embeddings.append(SparseEmbedding(name=key, indices=indices, values=values))
                    continue
                except (ValueError, TypeError):
                    pass

            # everything else goes to metadata
            metadata[key] = val

        return Chunk(
            id=entity["id"],
            text=entity.get("text", ""),
            embeddings=embeddings,
            metadata=metadata,
        )

    @staticmethod
    def _metric_from_config(cfg: VectorConfig) -> str:
        """Map VectorConfig.distance to Milvus metric."""
        v = (getattr(cfg.distance, "value", str(cfg.distance)) or "").upper()
        if "COS" in v:
            return "COSINE"
        if "IP" in v:
            return "IP"
        return "L2"

    @staticmethod
    def _sparse_to_milvus(sparse_emb: SparseEmbedding) -> dict[int, float]:
        """Convert SparseEmbedding to Milvus format."""
        return {i: v for i, v in zip(sparse_emb.indices, sparse_emb.values)}

    def add(
            self,
            chunk: Chunk | list[Chunk],
            collection_name: str,
            batch_size: int | None = None
    ):
        """
        Insert chunks into collection with batching.

        Args:
            chunk: Single chunk or list of chunks
            collection_name: Target collection name
            batch_size: Override default batch size
        """
        client = self.get_client()
        chunks = chunk if isinstance(chunk, list) else [chunk]
        batch_size = batch_size or self.batch_size

        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                rows = [self._chunk_to_row(c) for c in batch]
                client.insert(collection_name=collection_name, data=rows)
            log.info(f"Inserted {len(chunks)} chunks into '{collection_name}'")
        except MilvusException as e:
            log.error(f"Failed to insert into '{collection_name}': {e}")
            raise

    async def a_add(
            self,
            chunk: Chunk | list[Chunk],
            collection_name: str,
            batch_size: int | None = None
    ):
        """Async version of add()."""
        client = self._get_a_client()
        chunks = chunk if isinstance(chunk, list) else [chunk]
        batch_size = batch_size or self.batch_size

        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                rows = [self._chunk_to_row(c) for c in batch]
                await client.insert(collection_name=collection_name, data=rows)
            log.info(f"Inserted {len(chunks)} chunks into '{collection_name}'")
        except MilvusException as e:
            log.error(f"Failed to insert into '{collection_name}': {e}")
            raise

    def retrieve(
            self,
            collection_name: str,
            ids: list[str],
            output_fields: list[str] | None = None,
            **kwargs,
    ) -> list[Chunk]:
        """
        Retrieve chunks by IDs.

        Args:
            collection_name: Collection to query
            ids: List of chunk IDs
            output_fields: Fields to retrieve (None = all)
            **kwargs: Additional query parameters
        """
        client = self.get_client()

        if output_fields:
            output_fields = list(set(["id", "text"] + output_fields))

        try:
            res = client.query(
                collection_name=collection_name,
                ids=ids,
                output_fields=output_fields or ["*"],
                **kwargs,
            )
            return [self._entity_to_chunk(r) for r in (res or [])]
        except Exception as e:
            log.error(f"Failed to retrieve from '{collection_name}': {e}")
            raise

    def remove(self, collection_name: str, ids: list[str], **kwargs):
        """Delete chunks by IDs."""
        client = self.get_client()
        client.delete(collection_name=collection_name, ids=ids, **kwargs)
        log.info(f"Removed {len(ids)} chunks from '{collection_name}'")

    def update(self, collection_name: str, chunk: Chunk | list[Chunk], **kwargs):
        """
        Upsert chunks (native Milvus upsert operation).

        Args:
            collection_name: Target collection
            chunk: Chunk(s) to upsert
            **kwargs: Additional upsert parameters
        """
        client = self.get_client()
        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        data = [self._chunk_to_row(c) for c in chunks]
        client.upsert(collection_name=collection_name, data=data, **kwargs)
        log.info(f"Upserted {len(chunks)} chunks into '{collection_name}'")

    @staticmethod
    def _merge_output_fields(user_fields: list[str] | None) -> list[str]:
        """Ensure required fields are included in output."""
        required = ["id", "text"]
        if user_fields is None:
            return required[:]
        if not isinstance(user_fields, list):
            raise TypeError("output_fields must be a list[str]")
        return list(set(required + user_fields))

    def _normalize_query(
            self,
            v: list[float] | DenseEmbedding | SparseEmbedding,
            vector_name: str | None,
    ) -> tuple[list[float] | dict[int, float], str]:
        """Normalize query vector and extract field name."""
        if isinstance(v, (DenseEmbedding, SparseEmbedding)):
            vector_name = vector_name or v.name
            if not vector_name:
                raise ValueError("Embedding must have a name")

            if isinstance(v, SparseEmbedding):
                return self._sparse_to_milvus(v), vector_name
            return v.vector, vector_name

        if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
            raise TypeError("query_vector must be list[float], DenseEmbedding, or SparseEmbedding")

        if not vector_name:
            raise ValueError("vector_name must be provided for raw vector lists")

        return v, vector_name

    def search(
            self,
            collection_name: str,
            query_vector: list[float] | DenseEmbedding | SparseEmbedding,
            k: int = 10,
            vector_name: str | None = None,
            output_fields: list[str] | None = None,
            **kwargs,
    ) -> list[Chunk]:
        """
        Similarity search on a collection.

        Args:
            collection_name: Collection to search
            query_vector: Query vector/embedding
            k: Number of results
            vector_name: Vector field name (auto-detected from embedding)
            output_fields: Fields to return
            **kwargs: Additional search parameters (filter, etc.)
        """
        client = self.get_client()

        vector, field_name = self._normalize_query(query_vector, vector_name)
        fields = self._merge_output_fields(output_fields)

        res = client.search(
            collection_name=collection_name,
            data=[vector],
            anns_field=field_name,
            limit=k,
            output_fields=fields,
            **kwargs,
        )

        hits = res[0] if res else []
        chunks = []
        for h in hits:
            entity = h.get("entity") or {k: v for k, v in h.items() if k not in {"score", "distance"}}
            chunk = self._entity_to_chunk(entity)
            # Add score to metadata
            chunk.metadata["score"] = h.get("score") or h.get("distance")
            chunks.append(chunk)

        return chunks

    async def a_search(
            self,
            collection_name: str,
            query_vector: list[float] | DenseEmbedding | SparseEmbedding,
            k: int = 10,
            vector_name: str | None = None,
            output_fields: list[str] | None = None,
            **kwargs,
    ) -> list[Chunk]:
        """Async version of search()."""
        client = self._get_a_client()

        vector, field_name = self._normalize_query(query_vector, vector_name)
        fields = self._merge_output_fields(output_fields)

        res = await client.search(
            collection_name=collection_name,
            data=[vector],
            anns_field=field_name,
            limit=k,
            output_fields=fields,
            **kwargs,
        )

        hits = res[0] if res else []
        chunks = []
        for h in hits:
            entity = h.get("entity") or {k: v for k, v in h.items() if k not in {"score", "distance"}}
            chunk = self._entity_to_chunk(entity)
            chunk.metadata["score"] = h.get("score") or h.get("distance")
            chunks.append(chunk)

        return chunks

    def hybrid_search(
            self,
            collection_name: str,
            dense_vector: list[float] | DenseEmbedding | None = None,
            sparse_vector: SparseEmbedding | None = None,
            dense_weight: float = 0.5,
            k: int = 10,
            **kwargs,
    ) -> list[Chunk]:
        """
        Perform hybrid search with dense and sparse vectors.

        Args:
            collection_name: Collection to search
            dense_vector: Dense query vector
            sparse_vector: Sparse query vector
            dense_weight: Weight for dense results (sparse weight = 1 - dense_weight)
            k: Number of results
            **kwargs: Additional search parameters
        """
        if dense_vector is None and sparse_vector is None:
            raise ValueError("At least one of dense_vector or sparse_vector must be provided")

        results = {}

        if dense_vector is not None:
            dense_chunks = self.search(
                collection_name,
                dense_vector,
                k=k * 2,
                **kwargs
            )
            for chunk in dense_chunks:
                score = chunk.metadata.get("score", 0)
                results[chunk.id] = {
                    "chunk": chunk,
                    "dense_score": 1 / (1 + score),
                    "sparse_score": 0,
                }

        if sparse_vector is not None:
            sparse_chunks = self.search(
                collection_name,
                sparse_vector,
                k=k * 2,
                **kwargs
            )
            for chunk in sparse_chunks:
                score = chunk.metadata.get("score", 0)
                if chunk.id in results:
                    results[chunk.id]["sparse_score"] = score
                else:
                    results[chunk.id] = {
                        "chunk": chunk,
                        "dense_score": 0,
                        "sparse_score": score,
                    }

        for item in results.values():
            item["hybrid_score"] = (
                    dense_weight * item["dense_score"] +
                    (1 - dense_weight) * item["sparse_score"]
            )
            item["chunk"].metadata["hybrid_score"] = item["hybrid_score"]

        # sort and return top k
        sorted_results = sorted(results.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return [item["chunk"] for item in sorted_results[:k]]

    def get_collections(self) -> list[str]:
        """List all collections."""
        return self.get_client().list_collections()

    def create_collection(
            self,
            collection_name: str,
            vector_config: list[VectorConfig],
            index_params: IndexParams | None = None,
            **kwargs,
    ):
        """
        Create a collection with support for multiple vector fields.

        Args:
            collection_name: Name for the new collection
            vector_config: List of vector configurations
            index_params: Custom index parameters (auto-generated if None)
            **kwargs: Additional collection parameters
        """
        client = self.get_client()

        if client.has_collection(collection_name):
            log.warning(f"Collection '{collection_name}' already exists")
            return

        # schema collection
        fields: list[FieldSchema] = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        ]

        vector_names = set()
        for cfg in vector_config:
            if cfg.name in vector_names:
                raise ValueError(f"Duplicate vector name: {cfg.name}")
            vector_names.add(cfg.name)

            if cfg.format == EmbeddingFormat.DENSE:
                fields.append(FieldSchema(name=cfg.name, dtype=DataType.FLOAT_VECTOR, dim=cfg.dimensions))
            elif cfg.format == EmbeddingFormat.SPARSE:
                fields.append(FieldSchema(name=cfg.name, dtype=DataType.SPARSE_FLOAT_VECTOR))
            else:
                raise ValueError(f"Unsupported format: {cfg.format}")

        schema = CollectionSchema(fields, enable_dynamic_field=True)

        # build indexes
        if index_params is None:
            index_params = client.prepare_index_params()
            for cfg in vector_config:
                if cfg.format == EmbeddingFormat.DENSE:
                    index_params.add_index(
                        field_name=cfg.name,
                        index_type="AUTOINDEX",
                        metric_type=self._metric_from_config(cfg),
                    )
                elif cfg.format == EmbeddingFormat.SPARSE:
                    index_params.add_index(
                        field_name=cfg.name,
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP",
                    )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            **kwargs,
        )
        client.load_collection(collection_name)
        log.info(f"Created collection '{collection_name}' with {len(vector_config)} vector field(s)")

    def delete_collection(self, collection_name: str, **kwargs):
        """Drop a collection."""
        client = self.get_client()
        if client.has_collection(collection_name):
            client.drop_collection(collection_name, **kwargs)
            log.info(f"Deleted collection '{collection_name}'")

    def describe_collection(self, collection_name: str) -> dict:
        """Get collection schema and statistics."""
        client = self.get_client()
        desc = client.describe_collection(collection_name)
        stats = client.get_collection_stats(collection_name)
        return {
            "schema": desc,
            "stats": stats,
        }

    def dump_collection(
            self,
            collection_name: str,
            page_size: int = 100,
    ) -> Generator[Chunk, None, None]:
        """Stream all chunks from a collection."""
        client = self.get_client()
        offset = 0

        while True:
            res = client.query(
                collection_name=collection_name,
                filter="",
                output_fields=["*"],
                limit=page_size,
                offset=offset,
            )
            if not res:
                break

            for r in res:
                yield self._entity_to_chunk(r)

            if len(res) < page_size:
                break
            offset += page_size

    def prepare_index_params(self, **kwargs) -> IndexParams:
        """Create IndexParams instance for custom index configuration."""
        return self.get_client().prepare_index_params(**kwargs)

    def close(self):
        """Close client connections."""
        if hasattr(self, "client"):
            self.client.close()
            delattr(self, "client")
        if hasattr(self, "a_client"):
            delattr(self, "a_client")
        log.info("Closed Milvus connections")
