# vector_store/milvus_vectorstore.py
import logging
from typing import Any, Generator, Optional
import asyncio
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from datapizza.core.vectorstore import Vectorstore, VectorConfig
from datapizza.type import Chunk, DenseEmbedding

log = logging.getLogger(__name__)


class MilvusVectorstore(Vectorstore):
    """
    Datapizza-AI implementation of a Milvus vectorstore.
    Provides both sync and async interfaces.
    """

    def __init__(self, host: str, port: str, alias: str, **kwargs):
        """
        Initialize MilvusVectorstore client.

        Args:
            host (str): Milvus host.
            port (str): Milvus port (default "19530").
            alias (str): Connection alias.
            **kwargs: Additional kwargs passed to pymilvus.connections.connect.
        """
        self.host = host
        self.port = port
        self.connection_name = alias
        self.kwargs = kwargs
        self.client_connected = False
        self._connect()

    def _connect(self):
        """Connect to Milvus server if not already connected."""
        if not self.client_connected:
            connections.connect(alias=self.connection_name, host=self.host, port=self.port, **self.kwargs)
            log.info(f"Connected to Milvus at {self.host}:{self.port}")
            self.client_connected = True

    def get_client(self) -> Any:
        """Return the pymilvus client object (connections module)."""
        self._connect()
        return connections.get_connection(self.connection_name)

    def create_collection(self, collection_name: str, vector_config: list[VectorConfig]):
        """
        Create a collection with specified vector configuration.

        Args:
            collection_name (str): Name of the collection.
            vector_config (list[VectorConfig]): Vector configuration(s).
        """
        if utility.has_collection(collection_name):
            log.warning(f"Collection '{collection_name}' already exists, skipping creation.")
            return

        dim = vector_config[0].dimensions
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="Datapizza Milvus vector store")
        Collection(name=collection_name, schema=schema)
        log.info(f"Created Milvus collection '{collection_name}' with dim={dim}")

    def delete_collection(self, collection_name: str):
        """Delete a collection by name."""
        utility.drop_collection(collection_name)
        log.info(f"Collection '{collection_name}' deleted")

    def get_collections(self):
        """Return a list of collection names."""
        return utility.list_collections()

    def get_client(self):
        """Return the connection alias used for Milvus operations."""
        self._connect()
        return self.connection_name

    def load_collection(self, collection_name: str) -> Collection:
        """
        Load collection into memory for search.

        Args:
            collection_name (str): Name of collection.

        Returns:
            Collection: pymilvus Collection object.
        """
        collection = Collection(collection_name)
        collection.load()
        log.info(f"Collection '{collection_name}' loaded into memory")
        return collection

    def create_index(
            self,
            collection_name: str,
            field_name: str = "vector",
            index_type: str = "IVF_FLAT",
            metric_type: str = "L2",
            params: Optional[dict] = None,
    ):
        """
        Create an index on a collection field.

        Args:
            collection_name (str): Collection name.
            field_name (str): Field to index (default "vector").
            index_type (str): Index type (default "IVF_FLAT").
            metric_type (str): Metric type (default "L2").
            params (dict, optional): Index parameters.
        """
        collection = Collection(collection_name)
        if params is None:
            params = {"nlist": 128}

        existing_indexes = [idx.field_name for idx in collection.indexes]
        if field_name not in existing_indexes:
            collection.create_index(field_name,
                                    {"index_type": index_type, "metric_type": metric_type, "params": params})
            log.info(f"Index '{field_name}' created for collection '{collection_name}'")
        else:
            log.info(f"Index '{field_name}' already exists for collection '{collection_name}'")

    # -----------------------------
    # CRUD methods
    # -----------------------------
    def add(self, chunk: Chunk | list[Chunk], collection_name: str):
        """
        Add one or more chunks to a collection.

        Args:
            chunk (Chunk | list[Chunk]): Chunk(s) to add.
            collection_name (str): Target collection name.
        """
        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        collection = Collection(collection_name)

        ids, texts, vectors = [], [], []
        for c in chunks:
            if not c.embeddings:
                raise ValueError("Chunk must have an embedding")
            ids.append(int(c.id) if str(c.id).isdigit() else hash(c.id) % (10 ** 9))
            texts.append(c.text)
            vectors.append(c.embeddings[0].vector)

        collection.insert([ids, texts, vectors])
        collection.flush()
        log.info(f"Inserted {len(chunks)} chunks into collection '{collection_name}'")

    async def a_add(self, chunk: Chunk | list[Chunk], collection_name: str):
        """Async version of add."""
        await asyncio.to_thread(self.add, chunk, collection_name)

    def search(self, collection_name: str, query_vector: list[float], k: int = 10) -> list[Chunk]:
        """
        Search for similar vectors in a collection.

        Args:
            collection_name (str): Collection name.
            query_vector (list[float]): Query vector.
            k (int): Number of results (default 10).

        Returns:
            list[Chunk]: Matching chunks.
        """
        collection = Collection(collection_name)
        collection.load()
        results = collection.search(data=[query_vector], anns_field="vector", param={"metric_type": "L2"}, limit=k,
                                    output_fields=["text"])
        return [
            Chunk(id=str(hit.id), text=hit.entity.get("text"), metadata={},
                  embeddings=[DenseEmbedding(name="vector", vector=query_vector)])
            for hit in results[0]
        ]

    async def a_search(self, collection_name: str, query_vector: list[float], k: int = 10) -> list[Chunk]:
        """Async version of search."""
        return await asyncio.to_thread(self.search, collection_name, query_vector, k)

    def remove(self, collection_name: str, ids: list[str]):
        """
        Remove chunks by IDs.

        Args:
            collection_name (str): Collection name.
            ids (list[str]): Chunk IDs to remove.
        """
        collection = Collection(collection_name)
        expr = f"id in {ids}"
        collection.delete(expr)
        collection.flush()
        log.info(f"Deleted {len(ids)} items from collection '{collection_name}'")

    async def a_remove(self, collection_name: str, ids: list[str]):
        """Async version of remove."""
        await asyncio.to_thread(self.remove, collection_name, ids)

    def dump_collection(self, collection_name: str, page_size: int = 100) -> Generator[Chunk, None, None]:
        """
        Dump all chunks from a collection.

        Args:
            collection_name (str): Collection name.
            page_size (int): Number of points per page (default 100).

        Yields:
            Chunk: Chunks from collection.
        """
        collection = Collection(collection_name)
        offset = 0
        while True:
            res = collection.query(expr="", output_fields=["id", "text"], offset=offset, limit=page_size)
            if not res:
                break
            for r in res:
                yield Chunk(id=str(r["id"]), text=r["text"], metadata={})
            offset += page_size

    async def a_dump_collection(self, collection_name: str, page_size: int = 100) -> Generator[Chunk, None, None]:
        """Async version of dump_collection."""
        offset = 0
        while True:
            res = await asyncio.to_thread(lambda: list(self.dump_collection(collection_name, page_size)))
            if not res:
                break
            for chunk in res:
                yield chunk
            break

    def retrieve(self, collection_name: str, ids: list[str], **kwargs) -> list[Chunk]:
        """Retrieve chunks by ID."""
        collection = self._get_collection(collection_name)
        res = collection.query(expr=f"id in {ids}", output_fields=["text"])
        return [Chunk(id=str(r["id"]), text=r["text"], metadata={}) for r in res]

    def update(self, collection_name: str, payload: dict, points: list[int], **kwargs):
        """Update chunks by ID (Milvus requires delete+insert)."""
        log.warning("Update not supported natively in Milvus; requires delete+insert.")