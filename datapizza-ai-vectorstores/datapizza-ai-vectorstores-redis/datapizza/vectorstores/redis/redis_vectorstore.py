import logging
from typing import Any

from pydantic import BaseModel

import redis
import redis.asyncio as async_redis
from datapizza.core.vectorstore import VectorConfig, Vectorstore
from datapizza.type import Chunk, DenseEmbedding, EmbeddingFormat

log = logging.getLogger(__name__)


class RedisCollection(BaseModel):
    name: str
    config: list[VectorConfig]

    # TODO: add validation config should support only dense embeddings and config should have always a name


class RedisVectorstore(Vectorstore):
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, **kargs):
        self.client: redis.Redis
        self.conn_kwargs = {
            "host": host,
            "port": port,
            "db": db,
            "decode_responses": True,
            **kargs,
        }
        self.prefix = "datapizza:vectorstore"

    def get_client(self) -> redis.Redis:
        if not hasattr(self, "client"):
            self._init_client()
        return self.client

    def _get_a_client(self) -> async_redis.Redis:
        if not hasattr(self, "a_client"):
            self._init_a_client()
        return self.a_client

    def _init_client(self):
        self.client = redis.Redis(**self.conn_kwargs)

    def _init_a_client(self):
        self.a_client = async_redis.Redis(**self.conn_kwargs)

    def _meta_key(self, collection_name: str) -> str:
        """Return the key for collection metadata"""
        return f"{self.prefix}:meta:{collection_name}"

    def _data_key(self, collection_name: str) -> str:
        """Returns collection content key prefix"""
        return f"{self.prefix}:content:{collection_name}"

    def _elem_key(self, chunck_id: str, chunck_field: str):
        return f"{chunck_id}:{chunck_field}"

    def _collection_fields(self, collection_name: str) -> list[str]:
        client = self.get_client()
        collection = RedisCollection.model_validate(
            client.json().get(self._meta_key(collection_name))
        )
        return [config.name for config in collection.config if config.name is not None]

    def get_collections(self):
        client = self.get_client()
        pattern = self._meta_key("*")
        # collections = [key.decode("utf-8").replace(pattern[:-1], '') for key in client.scan_iter(pattern)]
        collections = list(client.scan_iter(pattern))
        return collections

    def exists(self, name: str) -> bool:
        client = self.get_client()
        return client.exists(name) > 0  # type: ignore

    def create_collection(self, collection_name: str, vector_config: list[VectorConfig]):
        client = self.get_client()
        k = self._meta_key(collection_name)
        # Redis supports only dense emdeddings so configs need to be filtered
        collection = RedisCollection(
            name=collection_name,
            config=[x for x in vector_config if x.format == EmbeddingFormat.DENSE],
        )
        if len(collection.config) == 0:
            raise ValueError("Redis vector store support only dense embeddings.")
        client.json().set(k, "$", collection.model_dump(mode="json"))

    def delete_collection(self, collection_name: str, **kargs):
        client = self.get_client()
        client.delete(self._data_key(collection_name))
        client.delete(self._meta_key(collection_name))

    def add(self, chunk: Chunk | list[Chunk], collection_name: str | None = None):
        client = self.get_client()
        if collection_name is None:
            raise ValueError("Collection name must be set")

        if not self.exists(self._meta_key(collection_name)):
            raise ValueError("Collection must be create to use it")

        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        client = client.vset()
        for item in chunks:
            attrs = {"id": item.id, "text": item.text, "metadata": item.metadata}
            for embedding in item.embeddings:
                key = self._data_key(collection_name)
                element = f"{item.id}:{embedding.name}"
                if not hasattr(embedding, "vector"):
                    raise ValueError("Redis vector store support only dense embeddings.")
                client.vadd(key, embedding.vector, element, attributes=attrs)  # type: ignore

    async def a_add(self, chunk: Chunk | list[Chunk], collection_name: str | None = None):
        client = self._get_a_client()
        if collection_name is None:
            raise ValueError("Collection name must be set")

        if not await client.exists(self._meta_key(collection_name)):
            raise ValueError("Collection must be create to use it")

        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        client = client.vset()
        for item in chunks:
            attrs = {"id": item.id, "text": item.text, "metadata": item.metadata}
            for embedding in item.embeddings:
                key = self._data_key(collection_name)
                element = f"{item.id}:{embedding.name}"
                if not hasattr(embedding, "vector"):
                    raise ValueError("Redis vector store support only dense embeddings.")
                await client.vadd(key, embedding.vector, element, attributes=attrs)  # type: ignore

    def update(self, collection_name: str, payload: dict, points: list[str], **kwargs):
        client = self.get_client()
        collection_key = self._data_key(collection_name)
        collection_fields = self._collection_fields(collection_name)
        for identifier in points:
            for field in collection_fields:
                element_key = self._elem_key(identifier, field)
                client.vset().vsetattr(collection_key, element_key, payload)
        return

    def remove(self, collection_name: str, ids: list[str], **kwargs):
        client = self.get_client()
        collection_key = self._data_key(collection_name)
        collection_fields = self._collection_fields(collection_name)
        for identifier in ids:
            for field_name in collection_fields:
                element = self._elem_key(identifier, field_name)
                client.vset().vrem(collection_key, element)
        return

    def retrieve(self, collection_name: str, ids: list[str], **kwargs) -> list[Chunk]:
        client = self.get_client()
        collection_key = self._data_key(collection_name)
        collection_fields = self._collection_fields(collection_name)
        chunks = []
        for identifier in ids:
            embeddings, attrs = [], None
            for field_name in collection_fields:
                element = self._elem_key(identifier, field_name)
                vector = client.vset().vemb(collection_key, element)
                embeddings.append(DenseEmbedding(field_name, vector=vector))  # type: ignore
                if attrs is None:
                    attrs = client.vset().vgetattr(collection_key, element)
            chunks.append(Chunk(**attrs, embeddings=embeddings))  # type: ignore
        return chunks

    def search(
        self,
        collection_name: str,
        query_vector: list[float] | DenseEmbedding,
        k: int = 10,
        vector_name: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        client = self.get_client()
        collection_key = self._data_key(collection_name)
        vector = query_vector.vector if isinstance(query_vector, DenseEmbedding) else query_vector
        results: dict[str, Any] = client.vset().vsim(
            collection_key, vector, with_attribs=True, count=k
        )  # type: ignore
        assert isinstance(results, dict), "Results by VSIM were not a dict!"
        parsed = {}
        for item_k, item_v in results.items():
            identifier, field = item_k.split(":")
            if identifier not in parsed:
                parsed[identifier] = Chunk(**item_v)
            vector: list[float] = client.vset().vemb(self._data_key(collection_name), item_k)  # type: ignore
            parsed[identifier].embeddings.append(DenseEmbedding(field, vector))
        return list(parsed.values())

    async def a_search(
        self,
        collection_name: str,
        query_vector: list[float] | DenseEmbedding,
        k: int = 10,
        vector_name: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        client = self._get_a_client()
        collection_key = self._data_key(collection_name)
        vector = query_vector.vector if isinstance(query_vector, DenseEmbedding) else query_vector
        results: dict[str, Any] = await client.vset().vsim(
            collection_key, vector, with_attribs=True, count=k
        )  # type: ignore
        assert isinstance(results, dict), "Results by VSIM were not a dict!"
        parsed = {}
        for item_k, item_v in results.items():
            identifier, field = item_k.split(":")
            if identifier not in parsed:
                parsed[identifier] = Chunk(**item_v)
            vector: list[float] = await client.vset().vemb(self._data_key(collection_name), item_k)  # type: ignore
            parsed[identifier].embeddings.append(DenseEmbedding(field, vector))
        return list(parsed.values())
