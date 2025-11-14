# MistralEmbedder

```python
pip install datapizza-ai-embedders-mistral
```


## Usage

```python
from datapizza.embedders.mistral import MistralEmbedder

embedder = MistralEmbedder(
    api_key=os.getenv("MISTRAL_API_KEY"),
    model_name="mistral-embed",
)

embedding = mistral_embedder.embed(text)

# Embed multiple texts
embeddings = embedder.embed(
    ["Hello world", "Another text"],
)
```

### Async Embedding

```python
import asyncio
from datapizza.embedders.mistral import MistralEmbedder

async def embed_async():
    embedder = MistralEmbedder(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model_name="mistral-embed",
    )

    text = "Async embedding example"
    embedding = await embedder.a_embed(text)

    return embedding

# Run async function
embedding = asyncio.run(embed_async())
```
