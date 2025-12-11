# Azure OpenAI Embedder

Azure OpenAI embedder implementation for the datapizza-ai framework.

## Installation

```bash
pip install datapizza-ai-embedders-azure
```

## Usage

```python
from datapizza.embedders.azure import AzureOpenAIEmbedder

# Basic usage
embedder = AzureOpenAIEmbedder(
    api_key="your-azure-openai-api-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    model_name="text-embedding-ada-002"
)

# With all parameters
embedder = AzureOpenAIEmbedder(
    api_key="your-azure-openai-api-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    model_name="text-embedding-ada-002",
    azure_deployment="your-deployment-name",
    api_version="2024-02-15-preview"
)

# Generate embeddings
embeddings = embedder.embed("Hello world")
multiple_embeddings = embedder.embed(["Hello", "World"])

# Async usage
embeddings = await embedder.a_embed("Hello world")
```

## Parameters

- `api_key` (str): Your Azure OpenAI API key
- `azure_endpoint` (str): Your Azure OpenAI endpoint URL
- `model_name` (str, optional): The embedding model name to use
- `azure_deployment` (str, optional): Your Azure deployment name
- `api_version` (str, optional): The API version to use

## Methods

- `embed(text: str | list[str], model_name: str | None = None)`: Generate embeddings synchronously
- `a_embed(text: str | list[str], model_name: str | None = None)`: Generate embeddings asynchronously

Both methods return a single embedding for a single text input, or a list of embeddings for multiple text inputs.
