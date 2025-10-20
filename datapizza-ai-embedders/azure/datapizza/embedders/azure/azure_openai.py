from datapizza.core.embedder import BaseEmbedder


class AzureOpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        api_key: str,
        azure_endpoint: str,
        model_name: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
    ):
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.api_version = api_version
        self.model_name = model_name

        self.client = None
        self.a_client = None

    def _set_client(self):
        from openai import AzureOpenAI

        if not self.client:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment,
            )

    def _set_a_client(self):
        from openai import AsyncAzureOpenAI

        if not self.a_client:
            self.a_client = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment,
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
    ) -> list[float] | list[list[float]]:
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        texts = [text] if isinstance(text, str) else text

        client = self._get_a_client()
        response = await client.embeddings.create(input=texts, model=model)

        embeddings = [embedding.embedding for embedding in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings
