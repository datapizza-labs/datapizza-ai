# Regolo Embedder

Regolo embedder implementation for datapizza-ai.

## Installation

```bash
pip install datapizza-ai-embedders-regolo
```

## Usage

```python
from datapizza.embedders.regolo import RegoloEmbedder

embedder = RegoloEmbedder(api_key="your-regolo-api-key")
embeddings = embedder.embed("Hello world", model_name="gte-Qwen2")
```

## Available Models

- `gte-Qwen2`
- `Qwen3-Embedding-8B`

## Configuration

The embedder uses the Regolo API endpoint at `https://api.regolo.ai/v1` by default.
You can override this by passing a custom `base_url` parameter.

