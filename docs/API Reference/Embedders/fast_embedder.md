# FastEmbedder


```python
pip install datapizza-ai-embedders-fastembedder
```

<!-- prettier-ignore -->
::: datapizza.embedders.fastembedder.FastEmbedder
    options:
        show_source: false


## Usage

```python
from datapizza.embedders.fastembedder import FastEmbedder

embedder = FastEmbedder(
    model_name="Qdrant/bm25",
    embedding_name="bm25_embeddings",
)

# Embed text (returns sparse embeddings)
embeddings = embedder.embed(["Hello world", "Another text"])
print(embeddings)
```

## Features

- Uses FastEmbed for efficient sparse text embeddings
- Local model execution (no API calls required)
- Configurable model caching directory
- Custom embedding naming
- Sparse embedding format for memory efficiency
- Both sync and async embedding support
