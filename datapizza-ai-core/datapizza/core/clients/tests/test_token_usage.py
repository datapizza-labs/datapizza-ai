from datapizza.core.clients.models import TokenUsage


def test_token_usage_model():
    token_usage = TokenUsage(
        prompt_tokens=100,
        completion_tokens=200,
        cached_tokens=300,
        thinking_tokens=400,
    )
    assert token_usage.prompt_tokens == 100
    assert token_usage.completion_tokens == 200
    assert token_usage.cached_tokens == 300
    assert token_usage.thinking_tokens == 400


def test_sum_token_usage():
    token_usage1 = TokenUsage(
        prompt_tokens=100,
        completion_tokens=200,
        cached_tokens=300,
        thinking_tokens=400,
    )
    token_usage2 = TokenUsage(
        prompt_tokens=50,
        completion_tokens=100,
        cached_tokens=150,
        thinking_tokens=200,
    )
    sum = token_usage1 + token_usage2
    assert sum.prompt_tokens == 150
    assert sum.completion_tokens == 300
    assert sum.cached_tokens == 450
    assert sum.thinking_tokens == 600
