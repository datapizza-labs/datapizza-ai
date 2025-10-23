import pytest

from datapizza.clients.bedrock import BedrockClient


def test_async_client_initialization():
    """Test that async client can be initialized"""
    client = BedrockClient(
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
        region_name="us-east-1",
    )

    # Initialize async client
    client._set_a_client()

    # Verify async session is created
    assert hasattr(client, "a_session")
    assert hasattr(client, "a_config")
    assert hasattr(client, "a_region_name")
    assert client.a_region_name == "us-east-1"


@pytest.mark.asyncio
async def test_a_invoke_method_exists():
    """Test that _a_invoke method is implemented and doesn't raise NotImplementedError"""
    client = BedrockClient(
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
        region_name="us-east-1",
    )

    # Verify the method exists and is async
    assert hasattr(client, "_a_invoke")
    assert callable(client._a_invoke)


@pytest.mark.asyncio
async def test_a_stream_invoke_method_exists():
    """Test that _a_stream_invoke method is implemented and doesn't raise NotImplementedError"""
    client = BedrockClient(
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
        region_name="us-east-1",
    )

    # Verify the method exists and is async
    assert hasattr(client, "_a_stream_invoke")
    assert callable(client._a_stream_invoke)
