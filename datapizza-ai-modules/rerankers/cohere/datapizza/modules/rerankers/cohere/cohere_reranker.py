from datapizza.core.modules.reranker import Reranker
from datapizza.type import Chunk


class CohereReranker(Reranker):
    """A reranker that uses the Cohere API to rerank documents."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        top_n: int = 10,
        threshold: float | None = None,
        model: str = "model",
    ):
        """
        Args:
            api_key: The API key for the Cohere API.
            endpoint: The endpoint for the Cohere API.
            top_n: The number of documents to return.
            threshold: The threshold for the reranker.
        """

        self.api_key = api_key
        self.endpoint = endpoint
        self.top_n = top_n
        self.threshold = threshold
        self.model = model

        self.client = None
        self.a_client = None

    def _set_client(self):
        import cohere

        if not self.client:
            self.client = cohere.ClientV2(base_url=self.endpoint, api_key=self.api_key)

    def _set_a_client(self):
        import cohere

        if not self.a_client:
            self.a_client = cohere.AsyncClientV2(
                base_url=self.endpoint,
                api_key=self.api_key,
            )

    def _get_client(self):
        if not self.client:
            self._set_client()

        if not self.client:
            raise RuntimeError("Client not set")

        return self.client

    def _get_a_client(self):
        if not self.a_client:
            self._set_a_client()

        if not self.a_client:
            raise RuntimeError("Client not set")

        return self.a_client

    def rerank(self, query: str, documents: list[Chunk]) -> list[Chunk]:
        """
        Rerank documents based on query.

        Args:
            query: The query to rerank documents by.
            documents: The documents to rerank.

        Returns:
            The reranked documents.
        """
        client = self._get_client()

        response = client.rerank(
            model=self.model,
            query=query,
            documents=[single_document.text for single_document in documents],
            top_n=self.top_n,
        )

        result_chunks: list[Chunk] = []

        for document in response.results:
            index = document.index
            relevance_score = document.relevance_score

            if self.threshold is not None and relevance_score < self.threshold:
                continue

            original_chunk = documents[index]
            updated_metadata = {
                **original_chunk.metadata,
                "reranker_score": relevance_score,
            }
            result_chunks.append(
                Chunk(
                    id=original_chunk.id,
                    text=original_chunk.text,
                    embeddings=original_chunk.embeddings,
                    metadata=updated_metadata,
                )
            )

        return result_chunks

    async def a_rerank(self, query: str, documents: list[Chunk]) -> list[Chunk]:
        """
        Rerank documents based on query.

        Args:
            query: The query to rerank documents by.
            documents: The documents to rerank.

        Returns:
            The reranked documents.
        """
        if not documents:
            return []

        client = self._get_a_client()

        response = await client.rerank(
            model=self.model,
            query=query,
            documents=[single_document.text for single_document in documents],
            top_n=self.top_n,
        )

        result_chunks: list[Chunk] = []

        for result in response.results:
            index = result.index
            score = result.relevance_score

            # Apply threshold filtering if specified
            if self.threshold is not None and score < self.threshold:
                continue

            original_chunk = documents[index]

            # Add reranker_score to metadata
            updated_metadata = {**original_chunk.metadata, "reranker_score": score}
            result_chunks.append(
                Chunk(
                    id=original_chunk.id,
                    text=original_chunk.text,
                    embeddings=original_chunk.embeddings,
                    metadata=updated_metadata,
                )
            )

        return result_chunks
