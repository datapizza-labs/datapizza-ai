import pytest
from unittest.mock import Mock, patch, AsyncMock
from datapizza.embedders.azure.azure_openai import AzureOpenAIEmbedder


def test_azure_openai_embedder_init():
    embedder = AzureOpenAIEmbedder(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com/",
        model_name="text-embedding-ada-002"
    )
    assert embedder.api_key == "test-key"
    assert embedder.azure_endpoint == "https://test.openai.azure.com/"
    assert embedder.model_name == "text-embedding-ada-002"
    assert embedder.azure_deployment is None
    assert embedder.api_version is None
    assert embedder.client is None
    assert embedder.a_client is None


def test_azure_openai_embedder_init_with_all_params():
    embedder = AzureOpenAIEmbedder(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com/",
        model_name="text-embedding-ada-002",
        azure_deployment="test-deployment",
        api_version="2024-02-15-preview"
    )
    assert embedder.api_key == "test-key"
    assert embedder.azure_endpoint == "https://test.openai.azure.com/"
    assert embedder.model_name == "text-embedding-ada-002"
    assert embedder.azure_deployment == "test-deployment"
    assert embedder.api_version == "2024-02-15-preview"


@patch('openai.AzureOpenAI')
def test_set_client(mock_azure_openai):
    embedder = AzureOpenAIEmbedder(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com/",
        model_name="text-embedding-ada-002",
        azure_deployment="test-deployment",
        api_version="2024-02-15-preview"
    )
    
    embedder._set_client()
    
    mock_azure_openai.assert_called_once_with(
        api_key="test-key",
        api_version="2024-02-15-preview",
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="test-deployment",
    )


@patch('openai.AsyncAzureOpenAI')
def test_set_a_client(mock_async_azure_openai):
    embedder = AzureOpenAIEmbedder(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com/",
        model_name="text-embedding-ada-002",
        azure_deployment="test-deployment",
        api_version="2024-02-15-preview"
    )
    
    embedder._set_a_client()
    
    mock_async_azure_openai.assert_called_once_with(
        api_key="test-key",
        api_version="2024-02-15-preview",
        azure_endpoint="https://test.openai.azure.com/",
        azure_deployment="test-deployment",
    )


@patch('openai.AzureOpenAI')
def test_embed_single_text(mock_azure_openai):
    # Mock the response
    mock_response = Mock()
    mock_embedding = Mock()
    mock_embedding.embedding = [0.1, 0.2, 0.3]
    mock_response.data = [mock_embedding]
    
    mock_client = Mock()
    mock_client.embeddings.create.return_value = mock_response
    mock_azure_openai.return_value = mock_client
    
    embedder = AzureOpenAIEmbedder(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com/",
        model_name="text-embedding-ada-002"
    )
    
    result = embedder.embed("Hello world")
    
    assert result == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once_with(
        input=["Hello world"], 
        model="text-embedding-ada-002"
    )


@patch('openai.AzureOpenAI')
def test_embed_multiple_texts(mock_azure_openai):
    # Mock the response
    mock_response = Mock()
    mock_embedding1 = Mock()
    mock_embedding1.embedding = [0.1, 0.2, 0.3]
    mock_embedding2 = Mock()
    mock_embedding2.embedding = [0.4, 0.5, 0.6]
    mock_response.data = [mock_embedding1, mock_embedding2]
    
    mock_client = Mock()
    mock_client.embeddings.create.return_value = mock_response
    mock_azure_openai.return_value = mock_client
    
    embedder = AzureOpenAIEmbedder(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com/",
        model_name="text-embedding-ada-002"
    )
    
    result = embedder.embed(["Hello", "World"])
    
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_client.embeddings.create.assert_called_once_with(
        input=["Hello", "World"], 
        model="text-embedding-ada-002"
    )


@patch('openai.AzureOpenAI')
def test_embed_with_custom_model_name(mock_azure_openai):
    # Mock the response
    mock_response = Mock()
    mock_embedding = Mock()
    mock_embedding.embedding = [0.1, 0.2, 0.3]
    mock_response.data = [mock_embedding]
    
    mock_client = Mock()
    mock_client.embeddings.create.return_value = mock_response
    mock_azure_openai.return_value = mock_client
    
    embedder = AzureOpenAIEmbedder(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com/",
        model_name="text-embedding-ada-002"
    )
    
    result = embedder.embed("Hello world", model_name="custom-model")
    
    assert result == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once_with(
        input=["Hello world"], 
        model="custom-model"
    )


def test_embed_without_model_name_raises_error():
    embedder = AzureOpenAIEmbedder(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com/"
    )
    
    with pytest.raises(ValueError, match="Model name is required."):
        embedder.embed("Hello world")


@patch('openai.AsyncAzureOpenAI')
@pytest.mark.asyncio
async def test_a_embed_single_text(mock_async_azure_openai):
    # Mock the response
    mock_response = Mock()
    mock_embedding = Mock()
    mock_embedding.embedding = [0.1, 0.2, 0.3]
    mock_response.data = [mock_embedding]
    
    mock_create = AsyncMock()
    mock_create.return_value = mock_response
    
    mock_client = Mock()
    mock_client.embeddings.create = mock_create
    mock_async_azure_openai.return_value = mock_client
    
    embedder = AzureOpenAIEmbedder(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com/",
        model_name="text-embedding-ada-002"
    )
    
    result = await embedder.a_embed("Hello world")
    
    assert result == [0.1, 0.2, 0.3]
    mock_create.assert_called_once_with(
        input=["Hello world"], 
        model="text-embedding-ada-002"
    )


@patch('openai.AsyncAzureOpenAI')
@pytest.mark.asyncio
async def test_a_embed_multiple_texts(mock_async_azure_openai):
    # Mock the response
    mock_response = Mock()
    mock_embedding1 = Mock()
    mock_embedding1.embedding = [0.1, 0.2, 0.3]
    mock_embedding2 = Mock()
    mock_embedding2.embedding = [0.4, 0.5, 0.6]
    mock_response.data = [mock_embedding1, mock_embedding2]
    
    mock_create = AsyncMock()
    mock_create.return_value = mock_response
    
    mock_client = Mock()
    mock_client.embeddings.create = mock_create
    mock_async_azure_openai.return_value = mock_client
    
    embedder = AzureOpenAIEmbedder(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com/",
        model_name="text-embedding-ada-002"
    )
    
    result = await embedder.a_embed(["Hello", "World"])
    
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_create.assert_called_once_with(
        input=["Hello", "World"], 
        model="text-embedding-ada-002"
    )


@pytest.mark.asyncio
async def test_a_embed_without_model_name_raises_error():
    embedder = AzureOpenAIEmbedder(
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com/"
    )
    
    with pytest.raises(ValueError, match="Model name is required."):
        await embedder.a_embed("Hello world")
