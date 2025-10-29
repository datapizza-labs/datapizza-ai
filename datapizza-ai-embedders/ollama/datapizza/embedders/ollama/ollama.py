from datapizza.core.embedder import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        model_name: str | None = None,
        base_url: str | None = None,
    ):
        self.base_url = base_url or "http://localhost:11434"
        self.model_name = model_name or "qwen3-embedding:8b"

        self.client = None
        self.a_client = None

    def _set_client(self):
        import ollama

        if not self.client:
            self.client = ollama.Client(host=self.base_url)

    def _set_a_client(self):
        import ollama

        if not self.a_client:
            self.a_client = ollama.AsyncClient(host=self.base_url)

    def embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float] | list[list[float]]:
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        texts = [text] if isinstance(text, str) else text

        client = self._get_client()

        embeddings = []
        for t in texts:
            response = client.embeddings(model=model, prompt=t)
            embeddings.append(response["embedding"])

        return embeddings[0] if isinstance(text, str) else embeddings

    async def a_embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float] | list[list[float]]:
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        texts = [text] if isinstance(text, str) else text

        client = self._get_a_client()

        embeddings = []
        for t in texts:
            response = await client.embeddings(model=model, prompt=t)
            embeddings.append(response["embedding"])

        return embeddings[0] if isinstance(text, str) else embeddings