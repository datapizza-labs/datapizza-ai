from types import SimpleNamespace

from datapizza.clients.google.google_client import GoogleClient


def test_google_token_usage_maps_thinking_tokens():
    client = object.__new__(GoogleClient)
    usage_metadata = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=20,
        cached_content_token_count=30,
        thoughts_token_count=40,
    )

    usage = client._token_usage_from_metadata(usage_metadata)

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.cached_tokens == 30
    assert usage.thinking_tokens == 40


def test_google_token_usage_uses_response_token_count_fallback():
    client = object.__new__(GoogleClient)
    usage_metadata = SimpleNamespace(
        prompt_token_count=10,
        response_token_count=25,
        cached_content_token_count=30,
    )

    usage = client._token_usage_from_metadata(usage_metadata)

    assert usage.completion_tokens == 25
    assert usage.thinking_tokens == 0
