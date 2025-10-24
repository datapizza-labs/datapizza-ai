import logging
from collections.abc import Generator

from datapizza.core.clients import Client
from datapizza.core.embedder import BaseEmbedder
from datapizza.core.models import PipelineComponent
from datapizza.type import Chunk, DenseEmbedding, SparseEmbedding

log = logging.getLogger(__name__)


class ClientEmbedder(BaseEmbedder):
    """Client embedder using any compatible LLM client with embedding capabilities."""

    def __init__(
        self,
        client: Client,
        model_name: str | None = None,
        embedding_name: str | None = None,
    ):
        self.client = client
        self.model_name = model_name
        self.embedding_name = embedding_name or model_name or self.client.model_name

    def embed(self, text: str | list[str], **kwargs) -> list[float]:
        return self.client.embed(text, self.model_name, **kwargs)  # type: ignore

    async def a_embed(self, text: str | list[str], **kwargs) -> list[float]:
        return await self.client.a_embed(text, self.model_name, **kwargs)  # type: ignore


class ChunkEmbedder(PipelineComponent):
    """ChunkEmbedder is a module that given a list of chunks, it put a list of embeddings in each chunk."""

    def __init__(
        self,
        client: Client,
        model_name: str | None = None,
        embedding_name: str | None = None,
        batch_size: int = 2047,
    ):
        """
        Initialize the ChunkEmbedder.

        Args:
            client (BaseEmbedder): The client to use for embedding.
            model_name (str, optional): The model name to use for embedding. Defaults to None.
            embedding_name (str, optional): The name of the embedding to use. Defaults to None.
            batch_size (int, optional): The batch size to use for embedding. Defaults to 2047.
        """
        self.client = client
        self.model_name = model_name
        self.embedding_name = embedding_name or model_name or self.client.model_name
        self.batch_size = batch_size

    def _batch_nodes(
        self, nodes: list[Chunk], batch_size: int
    ) -> Generator[list[Chunk], None, None]:
        for i in range(0, len(nodes), batch_size):
            yield nodes[i : i + batch_size]

    def embed(self, nodes: list[Chunk]) -> list[Chunk]:
        """
        Embeds the given list of chunks.

        Args:
            nodes (list[Chunk]): The list of chunks to embed.

        Returns:
            list[Chunk]: The list of chunks with embeddings.
        """
        if not all(isinstance(n, Chunk) for n in nodes):
            raise ValueError("Nodes must be of type Chunk")

        for batch in self._batch_nodes(nodes, batch_size=self.batch_size):
            embeddings = self.client.embed([n.text for n in batch], self.model_name)

            for n, embedding in zip(batch, embeddings, strict=False):
                if isinstance(embedding, SparseEmbedding):
                    embedding.name = self.embedding_name
                    n.embeddings.append(embedding)
                else:
                    n.embeddings.append(
                        DenseEmbedding(name=self.embedding_name, vector=embedding)  # type: ignore
                    )

        return nodes

    async def a_embed(self, nodes: list[Chunk]) -> list[Chunk]:
        """
        Asynchronously embeds the given list of chunks.

        Args:
            nodes (list[Chunk]): The list of chunks to embed.

        Returns:
            list[Chunk]: The list of chunks with embeddings.
        """
        if not all(isinstance(n, Chunk) for n in nodes):
            raise ValueError("Nodes must be of type Chunk")

        for batch in self._batch_nodes(nodes, batch_size=self.batch_size):
            embeddings = await self.client.a_embed(
                [n.text for n in batch], self.model_name
            )

            for n, embedding in zip(batch, embeddings, strict=False):
                if isinstance(embedding, SparseEmbedding):
                    n.embeddings.append(embedding)
                else:
                    n.embeddings.append(
                        DenseEmbedding(name=self.embedding_name, vector=embedding)  # type: ignore
                    )

        return nodes

    def _run(self, nodes: list[Chunk]) -> list[Chunk]:
        return self.embed(nodes)

    async def _a_run(self, nodes: list[Chunk]) -> list[Chunk]:
        return await self.a_embed(nodes)
