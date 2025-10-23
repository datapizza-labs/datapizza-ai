from datapizza.core.embedder import BaseEmbedder

from google import genai
from google.genai import types


class GoogleEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        api_key: str,
        model_name: str | None = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.task_type = task_type

        self.client = None
        self.a_client = None

    def _set_client(self):
        if not self.client:
            client = genai.Client(api_key=self.api_key)
            self.client = client

    def _set_a_client(self):
        if not self.a_client:
            client = genai.Client(api_key=self.api_key)
            self.a_client = client

    def embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float] | list[list[float]]:
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        texts = [text] if isinstance(text, str) else text

        client = self._get_client()

        result = client.models.embed_content(
            model=model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=self.task_type),
        )

        res = [embedding.values for embedding in result.embeddings]

        return res[0] if isinstance(text, str) else res

    async def a_embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float]:
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        texts = [text] if isinstance(text, str) else text

        client = self._get_a_client()

        result = await client.models.embed_content_async(
            model=model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=self.task_type),
        )

        res = [embedding.values for embedding in result.embeddings]

        return res[0] if isinstance(text, str) else res
