from datapizza.embedders.mistral import MistralEmbedder


def test_openai_embedder_init():
    embedder = MistralEmbedder(api_key="mistral-test-key")
    assert embedder.api_key == "mistral-test-key"
    assert embedder.base_url is None
    assert embedder.client is None
    assert embedder.a_client is None


def test_openai_embedder_init_with_base_url():
    embedder = MistralEmbedder(api_key="mistral-test-key", base_url="https://api.mistral.ai")
    assert embedder.api_key == "mistral-test-key"
    assert embedder.base_url == "https://api.mistral.ai"
