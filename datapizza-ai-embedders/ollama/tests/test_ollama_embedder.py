from datapizza.embedders.ollama import OllamaEmbedder


def test_ollama_embedder_init():
    embedder = OllamaEmbedder()
    assert embedder.model_name == "qwen3-embedding:8b"
    assert embedder.base_url == "http://localhost:11434"
    assert embedder.client is None
    assert embedder.a_client is None


def test_ollama_embedder_init_with_custom_model():
    embedder = OllamaEmbedder(model_name="nomic-embed-text")
    assert embedder.model_name == "nomic-embed-text"
    assert embedder.base_url == "http://localhost:11434"


def test_ollama_embedder_init_with_base_url():
    embedder = OllamaEmbedder(
        model_name="qwen3-embedding:8b",
        base_url="http://192.168.1.100:11434"
    )
    assert embedder.model_name == "qwen3-embedding:8b"
    assert embedder.base_url == "http://192.168.1.100:11434"