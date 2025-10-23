


# OllamaEmbedder

Ollama embedders are OpenAI-compatible, which means you can use the OpenAI embedder to generate embeddings with Ollama models. Simply configure the OpenAI embedder with Ollama's base URL and leave the API key empty.

## Usage


```python
from datapizza.embedders.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(
    api_key="",
    base_url="http://localhost:11434/v1",
)

# Embed a single text
embedding = embedder.embed("Hello world", model_name="nomic-embed-text")

print(embedding)

# Embed multiple texts
embeddings = embedder.embed(
    ["Hello world", "Another text"], model_name="nomic-embed-text"
)

print(embeddings)
```
