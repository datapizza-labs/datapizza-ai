from datapizza.core.embedder import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        api_key: str,
        model_name: str | None = None,
        base_url: str | None = None,
        input_type: str = "search_document",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

        self.input_type = input_type

        self.client = None
        self.a_client = None

    def _set_client(self):
        import cohere

        if not self.client:
            self.client = cohere.ClientV2(base_url=self.base_url, api_key=self.api_key)

    def _set_a_client(self):
        import cohere

        if not self.a_client:
            self.a_client = cohere.AsyncClientV2(
                base_url=self.base_url,
                api_key=self.api_key,
            )

    def embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float] | list[list[float]]:
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        texts = [text] if isinstance(text, str) else text

        client = self._get_client()

        response = client.embed(
            texts=texts,
            model=model,
            input_type=self.input_type,
            embedding_types=["float"],
        )
        embeddings = response.embeddings.float
        return embeddings[0] if isinstance(text, str) else embeddings

    async def a_embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float] | list[list[float]]:
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        texts = [text] if isinstance(text, str) else text

        client = self._get_a_client()
        response = await client.embed(
            texts=texts,
            model=model,
            input_type=self.input_type,
            embedding_types=["float"],
        )
        embeddings = response.embeddings.float
        return embeddings[0] if isinstance(text, str) else embeddings
