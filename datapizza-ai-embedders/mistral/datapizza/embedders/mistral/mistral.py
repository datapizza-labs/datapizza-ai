from datapizza.core.embedder import BaseEmbedder


class MistralEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model_name: str | None = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

        self.client = None
        self.a_client = None

    def _set_client(self) -> None:
        import mistralai

        if not self.client:
            self.client: mistralai.Mistral = mistralai.Mistral(api_key=self.api_key)

    def _set_a_client(self) -> None:
        import mistralai

        if not self.a_client:
            self.a_client: mistralai.Mistral = mistralai.Mistral(api_key=self.api_key)

    def embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float] | list[list[float]]:
        import mistralai
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        texts = [text] if isinstance(text, str) else text

        client: mistralai.Mistral = self._get_client()
        embedding_response: mistralai.EmbeddingResponse = client.embeddings.create(
            inputs=texts,
            model=model,
            server_url=self.base_url
        )

        embeddings = [embedding_response_data.embedding for embedding_response_data in embedding_response.data]
        return embeddings[0] if isinstance(text, str) else embeddings

    async def a_embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float] | list[list[float]]:
        import mistralai
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        texts = [text] if isinstance(text, str) else text

        client: mistralai.Mistral = self._get_a_client()
        embedding_response: mistralai.EmbeddingResponse  = await client.embeddings.create_async(
            inputs=texts,
            model=model,
            server_url=self.base_url
        )

        embeddings = [embedding_response_data.embedding for embedding_response_data in embedding_response.data]
        return embeddings[0] if isinstance(text, str) else embeddings


