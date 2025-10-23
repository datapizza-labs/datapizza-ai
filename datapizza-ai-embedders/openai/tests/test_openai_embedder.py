from datapizza.embedders.openai import OpenAIEmbedder


def test_openai_embedder_init():
    embedder = OpenAIEmbedder(api_key="test-key")
    assert embedder.api_key == "test-key"
    assert embedder.base_url is None
    assert embedder.client is None
    assert embedder.a_client is None


def test_openai_embedder_init_with_base_url():
    embedder = OpenAIEmbedder(api_key="test-key", base_url="https://api.openai.com/v1")
    assert embedder.api_key == "test-key"
    assert embedder.base_url == "https://api.openai.com/v1"
