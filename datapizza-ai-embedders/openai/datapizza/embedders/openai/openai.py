from datapizza.core.embedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        api_key: str,
        model_name: str | None = None,
        base_url: str | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

        self.client = None
        self.a_client = None

    def _set_client(self):
        import openai

        if not self.client:
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _set_a_client(self):
        import openai

        if not self.a_client:
            self.a_client = openai.AsyncOpenAI(
                api_key=self.api_key, base_url=self.base_url
            )

    def embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float] | list[list[float]]:
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        texts = [text] if isinstance(text, str) else text

        client = self._get_client()

        response = client.embeddings.create(input=texts, model=model)

        embeddings = [embedding.embedding for embedding in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings

    async def a_embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float]:
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        texts = [text] if isinstance(text, str) else text

        client = self._get_a_client()
        response = await client.embeddings.create(input=texts, model=model)

        embeddings = [embedding.embedding for embedding in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings
