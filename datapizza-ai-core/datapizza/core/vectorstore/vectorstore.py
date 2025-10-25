import logging
from abc import abstractmethod
from enum import Enum

from pydantic import BaseModel

from datapizza.core.models import ChainableProducer, PipelineComponent
from datapizza.type import Chunk, EmbeddingFormat

log = logging.getLogger(__name__)


class Distance(Enum):
    COSINE = "Cosine"
    EUCLIDEAN = "Euclidean"


class VectorConfig(BaseModel):
    name: str
    format: EmbeddingFormat = EmbeddingFormat.DENSE
    dimensions: int | None = None
    distance: Distance = Distance.COSINE

    def model_post_init(self, __context) -> None:
        if self.format == EmbeddingFormat.DENSE and self.dimensions is None:
            raise ValueError("Dimensions must be specified for dense embeddings")


class Vectorstore(ChainableProducer):
    """
    A class that can produce a vectorstore.
    If a Vectorstore is used as a node in a pipeline, it will produce a retriever.
    """

    @abstractmethod
    def add(self, chunk: Chunk | list[Chunk], collection_name: str | None = None):
        pass

    @abstractmethod
    async def a_add(
        self, chunk: Chunk | list[Chunk], collection_name: str | None = None
    ):
        pass

    @abstractmethod
    def update(self, collection_name: str, payload: dict, points: list[int], **kwargs):
        pass

    @abstractmethod
    def remove(self, collection_name: str, ids: list[str], **kwargs):
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        vector_name: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        pass

    @abstractmethod
    async def a_search(
        self,
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        vector_name: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        pass

    @abstractmethod
    def retrieve(self, collection_name: str, ids: list[str], **kwargs) -> list[Chunk]:
        pass

    def as_retriever(self, **kwargs):
        return Retriever(self, **kwargs)

    def _as_module_component(self, **kwargs):
        return self.as_retriever(**kwargs)


class Retriever(PipelineComponent):
    def __init__(self, vectorstore: Vectorstore, **kwargs):
        self.vectorstore: Vectorstore = vectorstore
        self.kwargs = kwargs

    def _run(
        self, collection_name: str, query_vector: list[float], k: int = 10, **kwargs
    ):
        return self.vectorstore.search(collection_name, query_vector, k, **kwargs)

    async def _a_run(
        self, collection_name: str, query_vector: list[float], k: int = 10, **kwargs
    ):
        return await self.vectorstore.a_search(
            collection_name=collection_name, query_vector=query_vector, k=k, **kwargs
        )
