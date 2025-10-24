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
from qdrant_client import AsyncQdrantClient, QdrantClient, models

log = logging.getLogger(__name__)


class QdrantVectorstore(Vectorstore):
    """
    datapizza-ai implementation of a Qdrant vectorstore.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int = 6333,
        api_key: str | None = None,
        **kwargs,
    ):
        """
        Initialize the QdrantVectorstore.

        Args:
            host (str, optional): The host to use for the Qdrant client. Defaults to None.
            port (int, optional): The port to use for the Qdrant client. Defaults to 6333.
            api_key (str, optional): The API key to use for the Qdrant client. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the Qdrant client.
        """
        if host is None and "location" not in kwargs:
            raise ValueError("Either host or location must be provided")

        self.client: QdrantClient
        self.a_client: AsyncQdrantClient
        self.batch_size: int = 100
        self.host: str | None = host
        self.port: int = port
        self.api_key: str | None = api_key
        self.kwargs: dict[str, Any] = kwargs

    def get_client(self) -> QdrantClient:
        if not hasattr(self, "client"):
            self._init_client()
        return self.client

    def _get_a_client(self) -> AsyncQdrantClient:
        if not hasattr(self, "a_client"):
            self._init_a_client()
        return self.a_client

    def _init_client(self):
        self.client = QdrantClient(
            host=self.host, port=self.port, api_key=self.api_key, **self.kwargs
        )

    def _init_a_client(self):
        self.a_client = AsyncQdrantClient(
            host=self.host, port=self.port, api_key=self.api_key, **self.kwargs
        )

    def add(self, chunk: Chunk | list[Chunk], collection_name: str | None = None):
        """Add a single chunk or list of chunks to the vectorstore.
        Args:
            chunk (Chunk | list[Chunk]): The chunk or list of chunks to add.
            collection_name (str, optional): The name of the collection to add the chunks to. Defaults to None.
        """
        client = self.get_client()

        if not collection_name:
            raise ValueError("Collection name must be set")

        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        points = []

        for chunk in chunks:
            points.append(self._process_chunk(chunk))

        # TODO: Process in batches
        for p in points:
            try:
                client.upsert(collection_name=collection_name, points=[p], wait=True)
            except Exception as e:
                log.error(f"Failed to add points to Qdrant: {e!s}")
                raise e

    async def a_add(
        self, chunk: Chunk | list[Chunk], collection_name: str | None = None
    ):
        client = self._get_a_client()

        if not collection_name:
            raise ValueError("Collection name must be set")

        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        points = []

        for chunk in chunks:
            points.append(self._process_chunk(chunk))

        # TODO: Process in batches
        for p in points:
            try:
                await client.upsert(
                    collection_name=collection_name, points=[p], wait=True
                )
            except Exception as e:
                log.error(f"Failed to add points to Qdrant: {e!s}")
                raise e

    def _process_chunk(self, chunk: Chunk) -> models.PointStruct:
        """Process a chunk into a Qdrant point."""
        if not chunk.embeddings:
            raise ValueError("Chunk must have an embedding")

        vector = {}
        for v in chunk.embeddings:
            if isinstance(v, DenseEmbedding):
                vector[v.name] = v.vector
            elif isinstance(v, SparseEmbedding):
                vector[v.name] = models.SparseVector(values=v.values, indices=v.indices)
            else:
                raise ValueError(f"Unsupported embedding type: {type(v)}")

        return models.PointStruct(
            id=str(chunk.id),
            payload={
                "text": chunk.text,
                **chunk.metadata,
            },
            vector=vector,  # type: ignore
        )

    def update(self, collection_name: str, payload: dict, points: list[int], **kwargs):
        client = self.get_client()
        client.overwrite_payload(
            collection_name=collection_name,
            payload=payload,
            points=points,  # type: ignore
            **kwargs,
        )

    def retrieve(self, collection_name: str, ids: list[str], **kwargs) -> list[Chunk]:
        """Retrieve chunks from a collection by their IDs.
        Args:
            collection_name (str): The name of the collection to retrieve the chunks from.
            ids (list[str]): The IDs of the chunks to retrieve.
            **kwargs: Additional keyword arguments to pass to the Qdrant client.
        Returns:
            list[Chunk]: The list of chunks retrieved from the collection.
        """
        client = self.get_client()
        return self._point_to_chunk(
            client.retrieve(
                collection_name=collection_name,
                ids=ids,
                **kwargs,
            )
        )

    def remove(self, collection_name: str, ids: list[str], **kwargs):
        """Remove chunks from a collection by their IDs.
        Args:
            collection_name (str): The name of the collection to remove the chunks from.
            ids (list[str]): The IDs of the chunks to remove.
            **kwargs: Additional keyword arguments to pass to the Qdrant client.
        """
        client = self.get_client()
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=ids,  # type: ignore
            ),
            **kwargs,
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float] | SparseEmbedding | dict,
        k: int = 10,
        vector_name: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        """
        Search for chunks in a collection by their query vector.

        Args:
            collection_name (str): The name of the collection to search in.
            query_vector (list[float]): The query vector to search for.
            k (int, optional): The number of results to return. Defaults to 10.
            vector_name (str, optional): The name of the vector to search for. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the Qdrant client.

        Returns:
            list[Chunk]: The list of chunks found in the collection.
        """
        client = self.get_client()

        if isinstance(query_vector, list) and all(
            isinstance(v, float) for v in query_vector
        ):
            if not vector_name:
                vectors_config = list(
                    client.get_collection(collection_name).config.params.vectors
                )
                if vectors_config and len(vectors_config) > 1:
                    raise ValueError(
                        f"Vector name not specified and multiple dense vectors are configured. Available vector names: {vectors_config}"
                    )
                vector_name = str(vectors_config[0])
            qry = (vector_name, query_vector) if vector_name else query_vector

        elif isinstance(query_vector, dict):
            indices = query_vector.get("indices", [])
            values = query_vector.get("values", [])
            qry = models.NamedSparseVector(
                name=vector_name or "default",
                vector=models.SparseVector(indices=indices, values=values),
            )

        elif isinstance(query_vector, SparseEmbedding):
            qry = models.NamedSparseVector(
                name=query_vector.name,
                vector=models.SparseVector(
                    indices=query_vector.indices, values=query_vector.values
                ),
            )
        else:
            raise ValueError(f"Unsupported query vector type: {type(query_vector)}")

        hits = client.search(
            collection_name=collection_name,
            query_vector=qry,
            limit=k,  # Return k closest points
            **kwargs,
        )
        return self._point_to_chunk(hits)

    async def a_search(
        self,
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        vector_name: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        """Search for chunks in a collection by their query vector."""
        client = self._get_a_client()

        qry = (vector_name, query_vector) if vector_name else query_vector

        hits = await client.search(
            collection_name=collection_name,
            query_vector=qry,
            limit=k,  # Return k closest points
            **kwargs,
        )
        return self._point_to_chunk(hits)

    def get_collections(self):
        """Get all collections in Qdrant."""
        client = self.get_client()
        return client.get_collections()

    def create_collection(
        self, collection_name: str, vector_config: list[VectorConfig], **kwargs
    ):
        """Create a new collection in Qdrant if it doesn't exist with the specified vector configurations

        Args:
            collection_name: Name of the collection to create
            vector_config: List of vector configurations specifying dimensions and distance metrics
            **kwargs: Additional arguments to pass to Qdrant's create_collection
        """

        client = self.get_client()

        if client.collection_exists(collection_name):
            log.warning(
                f"Collection {collection_name} already exists, skipping creation"
            )
            return

        sparse_config: (
            dict[str, models.SparseVectorParams] | models.SparseVectorParams | None
        ) = None
        config = None
        try:
            config = {
                v.name: models.VectorParams(
                    size=v.dimensions,  # type: ignore
                    distance=v.distance.value,  # type: ignore
                )
                for v in vector_config
                if v.format == EmbeddingFormat.DENSE
            }
            sparse_config = {
                v.name: models.SparseVectorParams()
                for v in vector_config
                if v.format == EmbeddingFormat.SPARSE
            }

            client.create_collection(
                collection_name=collection_name,
                vectors_config=config,
                sparse_vectors_config=sparse_config,
                **kwargs,
            )
        except Exception as e:
            log.error(f"Failed to create collection {collection_name}: {e!s}")
            raise e

    def delete_collection(self, collection_name: str, **kwargs):
        """Delete a collection in Qdrant."""
        client = self.get_client()
        client.delete_collection(collection_name=collection_name, **kwargs)

    def dump_collection(
        self,
        collection_name: str,
        page_size: int = 100,
        with_vectors: bool = False,
    ) -> Generator[Chunk, None, None]:
        """
        Dumps all points from a collection in a chunk-wise manner.

        Args:
            collection_name: Name of the collection to dump.
            page_size: Number of points to retrieve per batch.
            with_vectors: Whether to include vectors in the dumped chunks.

        Yields:
            Chunk: A chunk object from the collection.
        """
        client = self.get_client()
        next_page_offset = None

        while True:
            points, next_page_offset = client.scroll(
                collection_name=collection_name,
                limit=page_size,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=with_vectors,
            )

            if not points:
                break

            yield from self._point_to_chunk(points)

            if next_page_offset is None:
                break

    def _point_to_chunk(self, points) -> list[Chunk]:
        """
        Convert Qdrant points to Chunk objects.

        Args:
            points: List of Qdrant point objects

        Returns:
            List of Chunk objects with appropriate embeddings
        """
        chunks = []

        for point in points:
            vector = point.vector
            embeddings: list[Embedding] = []

            # Handle dictionary of named vectors
            if isinstance(vector, dict):
                for name, vec in vector.items():
                    if isinstance(vec, models.SparseVector):
                        embeddings.append(
                            SparseEmbedding(
                                name=name,
                                values=vec.values,
                                indices=vec.indices,
                            )
                        )
                    elif isinstance(vec, list):
                        embeddings.append(DenseEmbedding(name=name, vector=vec))
            # Handle single dense vector (list)
            elif isinstance(vector, list):
                embeddings.append(DenseEmbedding(name="dense", vector=vector))
            # Handle single sparse vector
            elif isinstance(vector, models.SparseVector):
                embeddings.append(
                    SparseEmbedding(
                        name="sparse", values=vector.values, indices=vector.indices
                    )
                )
            elif vector is None:
                embeddings = []
            else:
                raise ValueError(f"Unsupported vector type: {type(vector)}")

            chunks.append(
                Chunk(
                    id=point.id,
                    metadata=point.payload,
                    text=point.payload["text"],
                    embeddings=embeddings,
                )
            )

        return chunks
