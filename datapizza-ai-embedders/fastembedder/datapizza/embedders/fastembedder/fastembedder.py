import asyncio
import logging
from abc import abstractmethod

import fastembed
from datapizza.core.embedder import BaseEmbedder
from datapizza.type import DenseEmbedding, Embedding, SparseEmbedding

log = logging.getLogger(__name__)


class _BaseFastEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        embedding_name: str | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.embedding_name = embedding_name or model_name
        self.cache_dir = cache_dir
        self.embedder: fastembed.SparseTextEmbedding | fastembed.TextEmbedding

    @abstractmethod
    def _to_embedding(self, raw_embedding) -> Embedding:
        pass

    def embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> Embedding | list[Embedding]:
        embeddings = self.embedder.embed(text)
        results = [self._to_embedding(embedding) for embedding in embeddings]

        if isinstance(text, list):
            return results
        return results[0]

    async def a_embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> Embedding | list[Embedding]:
        return await asyncio.to_thread(self.embed, text)


class FastEmbedder(_BaseFastEmbedder):
    def __init__(
        self,
        model_name: str,
        embedding_name: str | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name, embedding_name, cache_dir, **kwargs)
        self.embedder = fastembed.SparseTextEmbedding(
            model_name=model_name, cache_dir=cache_dir, **kwargs
        )

    def _to_embedding(self, raw_embedding) -> SparseEmbedding:
        return SparseEmbedding(
            name=self.embedding_name,
            values=raw_embedding.values.tolist(),
            indices=raw_embedding.indices.tolist(),
        )


class FastDenseEmbedder(_BaseFastEmbedder):
    def __init__(
        self,
        model_name: str,
        embedding_name: str | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name, embedding_name, cache_dir, **kwargs)
        self.embedder = fastembed.TextEmbedding(
            model_name=model_name, cache_dir=cache_dir, **kwargs
        )

    def _to_embedding(self, raw_embedding) -> DenseEmbedding:
        return DenseEmbedding(
            name=self.embedding_name,
            vector=raw_embedding.tolist(),
        )
