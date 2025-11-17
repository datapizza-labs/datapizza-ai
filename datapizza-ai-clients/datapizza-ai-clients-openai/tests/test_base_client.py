from unittest.mock import MagicMock, patch

import httpx
from datapizza.core.clients import ClientResponse

from datapizza.clients.openai import (
    OpenAIClient,
)


def test_client_init():
    client = OpenAIClient(
        model="gpt-4o-mini",
        api_key="test_api_key",
    )
    assert client is not None


@patch("datapizza.clients.openai.openai_client.OpenAI")
def test_client_init_with_extra_args(mock_openai):
    """Tests that extra arguments are passed to the OpenAI client."""
    OpenAIClient(
        api_key="test_api_key",
        organization="test-org",
        project="test-project",
        timeout=30.0,
        max_retries=3,
    )
    mock_openai.assert_called_once_with(
        api_key="test_api_key",
        base_url=None,
        organization="test-org",
        project="test-project",
        webhook_secret=None,
        websocket_base_url=None,
        timeout=30.0,
        max_retries=3,
        default_headers=None,
        default_query=None,
        http_client=None,
    )


@patch("datapizza.clients.openai.openai_client.OpenAI")
def test_client_init_with_http_client(mock_openai):
    """Tests that a custom http_client is passed to the OpenAI client."""
    custom_http_client = httpx.Client()
    OpenAIClient(
        api_key="test_api_key",
        http_client=custom_http_client,
    )
    mock_openai.assert_called_once_with(
        api_key="test_api_key",
        base_url=None,
        organization=None,
        project=None,
        webhook_secret=None,
        websocket_base_url=None,
        timeout=None,
        max_retries=2,
        default_headers=None,
        default_query=None,
        http_client=custom_http_client,
    )


@patch("datapizza.clients.openai.openai_client.OpenAI")
def test_invoke_kwargs_override(mock_openai_class):
    """
    Tests that kwargs like 'stream' are not overridden by user input
    in non-streaming methods, but other kwargs are passed through.
    """
    mock_openai_instance = mock_openai_class.return_value
    mock_openai_instance.responses.create.return_value = MagicMock()

    client = OpenAIClient(api_key="test")
    client._response_to_client_response = MagicMock(
        return_value=ClientResponse(content=[])
    )

    client.invoke("hello", stream=True, top_p=0.5)

    mock_openai_instance.responses.create.assert_called_once()
    called_kwargs = mock_openai_instance.responses.create.call_args.kwargs

    assert called_kwargs.get("top_p") == 0.5
    assert called_kwargs.get("stream") is False


@patch("datapizza.clients.openai.openai_client.OpenAI")
def test_stream_invoke_kwargs_override(mock_openai_class):
    """
    Tests that kwargs like 'stream' are not overridden by user input
    in streaming methods.
    """
    mock_openai_instance = mock_openai_class.return_value
    mock_openai_instance.responses.create.return_value = []

    client = OpenAIClient(api_key="test")

    list(client.stream_invoke("hello", stream=False, top_p=0.5))

    mock_openai_instance.responses.create.assert_called_once()
    called_kwargs = mock_openai_instance.responses.create.call_args.kwargs

    assert called_kwargs.get("top_p") == 0.5
    assert called_kwargs.get("stream") is True
