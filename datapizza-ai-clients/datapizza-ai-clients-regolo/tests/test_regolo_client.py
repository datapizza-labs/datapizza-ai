from datapizza.clients.regolo import RegoloClient


def test_regolo_init_and_base_url():
    client = RegoloClient(
        api_key="test_api_key",
        model="Llama-3.1-8B-Instruct",
        system_prompt="Assistant.",
    )
    assert client is not None
    assert str(client.base_url) == "https://api.regolo.ai/v1"
