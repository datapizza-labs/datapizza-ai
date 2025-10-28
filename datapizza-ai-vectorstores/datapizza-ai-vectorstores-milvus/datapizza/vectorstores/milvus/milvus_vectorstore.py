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
    MilvusException,
)
from pymilvus.milvus_client.index import IndexParams

log = logging.getLogger(__name__)


class MilvusVectorstore(Vectorstore):
    """
    Milvus Vectorstore
    """

    def __init__(
        self,
        # You can pass either `uri="http://localhost:19530"` | "./milvus.db" (Milvus Lite)
        # or host/port(+secure/user/password) via **connection_args for flexibility.
        uri: str | None = None,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        secure: bool | None = None,
        **connection_args: Any,
    ):
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
        self.client = MilvusClient(**self.conn_kwargs)

    def _init_a_client(self):
        self.a_client = AsyncMilvusClient(**self.conn_kwargs)

    @staticmethod
    def _chunk_to_row(chunk: Chunk) -> dict[str, Any]:
        def _sparse_to_dict(se: SparseEmbedding) -> dict[int, float]:
            return {
                int(i): float(v) for i, v in zip(se.indices, se.values, strict=False)
            }

        if not chunk.embeddings:
            raise ValueError("Chunk must have an embedding")

        row: dict[str, Any] = {
            "id": chunk.id,
            "text": chunk.text or "",
        }
        if chunk.metadata:
            row.update(chunk.metadata)

        if len(chunk.embeddings) == 1:
            e = chunk.embeddings[0]
            if isinstance(e, DenseEmbedding):
                row[e.name] = e.vector
            elif isinstance(e, SparseEmbedding):
                row[e.name] = _sparse_to_dict(e)
            else:
                raise ValueError(f"Unsupported embedding type: {type(e)}")
        else:
            for e in chunk.embeddings:
                if isinstance(e, DenseEmbedding):
                    fname = e.name
                    row[fname] = e.vector
                elif isinstance(e, SparseEmbedding):
                    fname = e.name
                    row[fname] = _sparse_to_dict(e)
                else:
                    raise ValueError(f"Unsupported embedding type: {type(e)}")
        return row

    @staticmethod
    def _entity_to_chunk(entity: dict[str, Any]) -> Chunk:
        """
        Convert a Milvus entity dict into a Chunk.
        - Dense vectors: list[float] -> DenseEmbedding(name=<field>)
        - Sparse vectors: dict[index->value] -> SparseEmbedding(name=<field>)
        - Everything else (except id/text) -> metadata
        """

        embeddings: list[Embedding] = []
        metadata: dict[str, Any] = {}

        for key, val in entity.items():
            if key in {"id", "text"}:
                continue

            if isinstance(val, list) and all(isinstance(x, float) for x in val):
                embeddings.append(DenseEmbedding(name=key, vector=list(val)))
                continue

            if isinstance(val, dict) and val:
                try:
                    items = sorted(
                        ((int(k), float(v)) for k, v in val.items()), key=lambda t: t[0]
                    )
                    indices = [i for i, _ in items]
                    values = [v for _, v in items]
                    embeddings.append(
                        SparseEmbedding(name=key, indices=indices, values=values)
                    )
                    continue
                except Exception:
                    # fall through to metadata if it isn't a proper sparse map
                    pass

            # Non-vector -> metadata
            metadata[key] = val

        return Chunk(
            id=entity["id"],
            text=entity["text"],
            embeddings=embeddings,
            metadata=metadata,
        )

    @staticmethod
    def _metric_from_config(cfg: VectorConfig) -> str:
        """Map your VectorConfig.distance to Milvus metric string."""
        v = (getattr(cfg.distance, "value", str(cfg.distance)) or "").upper()
        if "COS" in v:
            return "COSINE"
        return "L2"

    @staticmethod
    def _sparse_embedding_to_milvus_format(
        sparse_emb: SparseEmbedding,
    ) -> dict[int, float]:
        """
        Convert a SparseEmbedding instance to a Milvus-supported sparse vector format (dict).

        Milvus accepts: {dimension_index: value, ...}
        """
        return dict(zip(sparse_emb.indices, sparse_emb.values, strict=False))

    def add(
        self,
        chunk: Chunk | list[Chunk],
        collection_name: str | None = None,
        batch_size: int = 100,
    ):
        client = self.get_client()

        chunks = chunk if isinstance(chunk, list) else [chunk]
        total = len(chunks)

        try:
            for i in range(0, total, batch_size):
                batch = chunks[i : i + batch_size]
                rows = [self._chunk_to_row(c) for c in batch]
                client.insert(collection_name=collection_name, data=rows)

        except MilvusException as e:
            log.error(f"Failed to batch insert into '{collection_name}': {e!s}")
            raise

    async def a_add(
        self,
        chunk: Chunk | list[Chunk],
        collection_name: str | None = None,
        batch_size: int = 100,
    ):
        client = self._get_a_client()

        chunks = chunk if isinstance(chunk, list) else [chunk]
        total = len(chunks)

        try:
            for i in range(0, total, batch_size):
                batch = chunks[i : i + batch_size]
                rows = [self._chunk_to_row(c) for c in batch]
                await client.insert(collection_name=collection_name, data=rows)

        except MilvusException as e:
            log.error(f"Failed to batch insert into '{collection_name}': {e!s}")
            raise

    def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        **kwargs,
    ) -> list[Chunk]:
        """
        Retrieve chunks from a collection by their IDs.

        Args:
            collection_name (str): Name of the collection to query.
            ids (list[str]): IDs to retrieve.
            **kwargs: Extra arguments to Milvus client query().

        Returns:
            list[Chunk]: Retrieved chunks.
        """

        client = self.get_client()
        try:
            res = client.query(
                collection_name=collection_name,
                ids=ids,
                **kwargs,
            )
        except Exception as e:
            log.error(f"Failed to retrieve from collection '{collection_name}': {e!s}")
            raise

        return [self._entity_to_chunk(r) for r in (res or [])]

    def remove(self, collection_name: str, ids: list[str], **kwargs):
        client = self.get_client()
        client.delete(collection_name=collection_name, ids=ids, **kwargs)

    def update(self, collection_name: str, chunk: Chunk | list[Chunk], **kwargs):
        """
        Upsert one or more Chunk objects into a Milvus collection.

        In Milvus, an upsert operation combines both insert and update behavior:
          - If an entity with the same primary key already exists in the collection,
            it will be overwritten with the new data.
          - If the primary key does not exist, a new entity will be inserted.

        Args:
            collection_name (str): Name of the Milvus collection.
            chunk (Chunk | list[Chunk]): One or more Chunk objects to upsert.
            **kwargs: Additional parameters passed to the Milvus client.

        """
        client = self.get_client()
        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        data = [self._chunk_to_row(c) for c in chunks]
        client.upsert(collection_name=collection_name, data=data, **kwargs)
        client.flush(collection_name=collection_name)

    @staticmethod
    def _merge_output_fields(user_fields: list[str] | None) -> list[str]:
        required = ["id", "text"]
        if user_fields is None:
            return required[:]
        if not isinstance(user_fields, list) or not all(
            isinstance(f, str) for f in user_fields
        ):
            raise TypeError("output_fields must be a list[str]")
        # ensure required fields are included
        return required[:] + [f for f in user_fields if f not in required]

    @staticmethod
    def _is_multi_query(vec) -> bool:
        return (
            isinstance(vec, list)
            and vec
            and isinstance(vec[0], (list, DenseEmbedding, SparseEmbedding))
        )

    def _normalize_query(
        self,
        v: list[float] | DenseEmbedding | SparseEmbedding,
        vector_name: str | None,
    ) -> tuple[list[float] | dict[int, float], str | None]:
        if isinstance(v, (DenseEmbedding, SparseEmbedding)) and not vector_name:
            vector_name = v.name
        if isinstance(v, SparseEmbedding):
            return self._sparse_embedding_to_milvus_format(v), vector_name
        if isinstance(v, DenseEmbedding):
            return v.vector, vector_name
        if not isinstance(v, list) or (v and not isinstance(v[0], (int, float))):
            raise TypeError(
                "query_vector must be list[float], DenseEmbedding, or SparseEmbedding"
            )
        return v, vector_name

    def _prepare_search_args(
        self,
        *,
        query_vector: list[float] | DenseEmbedding | SparseEmbedding,
        vector_name: str | None,
        k: int,
        kwargs: dict,
    ) -> dict:
        # guard against accidental multi-vector input
        if self._is_multi_query(query_vector):
            raise TypeError(
                "Single-vector search only: got a list of vectors. Pass exactly one vector."
            )

        # normalize
        vector, vector_name = self._normalize_query(query_vector, vector_name)
        if not vector_name:
            raise ValueError(
                "vector_name must be provided (or embedded in the Embedding.name)."
            )

        # fields
        user_fields = kwargs.pop("output_fields", None)
        # ensure at least id and text are included
        output_fields = self._merge_output_fields(user_fields)

        # final param bundle for Milvus .search()
        return {
            "data": [vector],
            "anns_field": vector_name,
            "limit": k,
            "output_fields": output_fields,
            **kwargs,
        }

    def _chunks_from_results(self, res) -> list[Chunk]:
        hits = res[0] if res else []
        out: list[Chunk] = []
        for h in hits:
            entity = h.get("entity", None)
            if entity is None:
                entity = {k: v for k, v in h.items() if k not in {"score", "distance"}}
            out.append(self._entity_to_chunk(entity))
        return out

    def search(
        self,
        collection_name: str,
        query_vector: list[float] | DenseEmbedding | SparseEmbedding,
        k: int = 10,
        vector_name: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        """
        Perform a single-vector similarity search on a Milvus collection.

        Args:
            collection_name (str): Name of the Milvus collection to search.
            query_vector (list[float] | DenseEmbedding | SparseEmbedding):
                The query vector or embedding to search for.
            k (int, optional): Number of nearest results to return. Defaults to 10.
            vector_name (str, optional): Name of the vector field to search on.
                If not provided, inferred from the embedding (if applicable).
            **kwargs: Additional parameters passed directly to the Milvus client's
                `search()` method (e.g., filter expressions, consistency level).

        Returns:
            list[Chunk]: A list of retrieved `Chunk` objects representing the top
            `k` most similar results.
        """
        client = self.get_client()
        params = self._prepare_search_args(
            query_vector=query_vector,
            vector_name=vector_name,
            k=k,
            kwargs=kwargs,
        )
        res = client.search(collection_name=collection_name, **params)
        return self._chunks_from_results(res)

    # ---------- async API ----------

    async def a_search(
        self,
        collection_name: str,
        query_vector: list[float] | DenseEmbedding | SparseEmbedding,
        k: int = 10,
        vector_name: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        """
        Perform an asynchronous single-vector similarity search on a Milvus collection.

        Args:
            collection_name (str): Name of the Milvus collection to search.
            query_vector (list[float] | DenseEmbedding | SparseEmbedding):
                The query vector or embedding to search for.
            k (int, optional): Number of nearest results to return. Defaults to 10.
            vector_name (str, optional): Name of the vector field to search on.
                If not provided, inferred from the embedding (if applicable).
            **kwargs: Additional parameters passed directly to the Milvus client's
                `search()` method (e.g., filter expressions, consistency level).

        Returns:
            list[Chunk]: A list of retrieved `Chunk` objects representing the top
            `k` most similar results.
        """
        client = self._get_a_client()
        params = self._prepare_search_args(
            query_vector=query_vector,
            vector_name=vector_name,
            k=k,
            kwargs=kwargs,
        )
        res = await client.search(collection_name=collection_name, **params)
        return self._chunks_from_results(res)

    def get_collections(self):
        client = self.get_client()
        return client.list_collections()

    def create_collection(
        self,
        collection_name: str,
        vector_config: list[VectorConfig],
        index_params: IndexParams = None,
        **kwargs,
    ):
        """
        Create a collection with:
          - id (VARCHAR primary key), text (VARCHAR)
          - one or more vector fields from `vector_config`
          - dynamic fields enabled (so `chunk.metadata` will be stored in $meta)
          - per-field indexes with metrics derived from `VectorConfig.distance`
        """

        client = self.get_client()
        if client.has_collection(collection_name):
            log.warning(
                f"Collection {collection_name} already exists, skipping creation"
            )
            return

        fields: list[FieldSchema] = [
            # Fixed 36-char UUIDv4 string (includes hyphens), used as primary key
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        ]

        # Build vector fields
        vector_names = set()
        for cfg in vector_config:
            if cfg.format == EmbeddingFormat.DENSE:
                fname = cfg.name
                if fname in vector_names:
                    raise ValueError(
                        "Each VectorConfig must have a unique 'name' for multi-vector collections"
                    )
                vector_names.add(fname)
                fields.append(
                    FieldSchema(
                        name=fname, dtype=DataType.FLOAT_VECTOR, dim=cfg.dimensions
                    )
                )
            elif cfg.format == EmbeddingFormat.SPARSE:
                fname = cfg.name
                if fname in vector_names:
                    raise ValueError(
                        "Each VectorConfig must have a unique 'name' for multi-vector collections"
                    )
                vector_names.add(fname)
                fields.append(
                    FieldSchema(
                        name=fname,
                        dtype=DataType.SPARSE_FLOAT_VECTOR,
                    )
                )
            else:
                raise ValueError(f"Unsupported embedding format: {cfg.format}")

        # Create the schema for the collection
        # TODO: add support for additional kwargs
        #  to pass to schema e.g. partition key, Milvus built in functions
        schema = CollectionSchema(
            fields,
            # default true for flexibility, Chunk's dynamic metadata will be stored into $meta
            # https://milvus.io/docs/enable-dynamic-field.md#Dynamic-Field
            enable_dynamic_field=True,
        )

        idx_params = client.prepare_index_params()
        if index_params is not None:
            idx_params = index_params
        # Add indexes
        if index_params is None:
            for cfg in vector_config:
                if cfg.format == EmbeddingFormat.DENSE:
                    fname = cfg.name
                    default_index_params = {
                        "field_name": fname,
                        "index_type": "AUTOINDEX",
                        "metric_type": self._metric_from_config(cfg),
                        "params": {},
                    }
                    idx_params.add_index(**default_index_params)
                elif cfg.format == EmbeddingFormat.SPARSE:
                    fname = cfg.name
                    default_index_params = {
                        "field_name": fname,
                        "index_type": "SPARSE_INVERTED_INDEX",
                        "metric_type": "IP",  # cosine not supported
                        "params": {},
                    }
                    idx_params.add_index(**default_index_params)

        # Create Collection
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=idx_params,
            **kwargs,
        )
        # load the collection by default
        client.load_collection(collection_name)

    def delete_collection(self, collection_name: str, **kwargs):
        client = self.get_client()
        if client.has_collection(collection_name):
            client.drop_collection(collection_name, **kwargs)

    def dump_collection(
        self,
        collection_name: str,
        page_size: int = 100,
    ) -> Generator[Chunk, None, None]:
        """
        Iterate all rows via offset paging.
        """
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
        client = self.get_client()
        return client.prepare_index_params(**kwargs)
