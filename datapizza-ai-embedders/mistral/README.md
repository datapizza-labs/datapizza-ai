# Google Embedder

Mistral AI embedder implementation for datapizza-ai

## Installation

```bash
pip install datapizza-ai-embedders-mistral
```

## Usage

```python
from datapizza.embedders.mistral  import MistralEmbedder

embedder = MistralEmbedder(api_key="your-google-api-key")
embeddings = embedder.embed("Hello world", model_name="models/text-embedding-004")
```
