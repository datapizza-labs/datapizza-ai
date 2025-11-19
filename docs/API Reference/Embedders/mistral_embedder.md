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

## Ingestion

Set the batch size to a conservative value (eg: 150) to avoid the error `Too many inputs in request`:


```python
                ingestion_pipeline = IngestionPipeline(
                    modules=[
                        # some modules
                        ChunkEmbedder(
                            client=MistralEmbedder(
                                api_key="some_key",
                                model_name="mistral-embed",
                            ),
                            batch_size=150  # Mistral API limit (conservative)
                        ),  
                    ],
                )
```

### The full error
```json
{
  "object": "error",
  "message": "Too many inputs in request, split into more batches.",
  "type": "invalid_request_prompt",
  "param": null,
  "code": "3210"
}
```
