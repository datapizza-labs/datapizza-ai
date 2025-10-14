from datapizza.embedders.regolo import RegoloEmbedder


def test_regolo_embedder_init():
    embedder = RegoloEmbedder(api_key="test-key")
    assert embedder.api_key == "test-key"
    assert embedder.base_url == "https://api.regolo.ai/v1"
    assert embedder.client is None
    assert embedder.a_client is None


def test_regolo_embedder_with_model():
    embedder = RegoloEmbedder(api_key="test-key", model_name="gte-Qwen2")
    assert embedder.model_name == "gte-Qwen2"

