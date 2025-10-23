# Rerankers

Rerankers are pipeline components that reorder and score retrieved content based on relevance to a query. They improve retrieval quality by applying more sophisticated ranking algorithms after initial retrieval, helping surface the most relevant content for user queries.

## Installation

All rerankers require separate installation via pip and are not included by default with `datapizza-ai-core`.

## Available Rerankers

### Optional Rerankers (Separate Installation Required)

- [CohereReranker](cohere_reranker.md) - Uses Cohere's reranking API for high-quality semantic reranking
- [TogetherReranker](together_reranker.md) - Uses Together AI's API with various model options

## Common Features

- High-quality semantic reranking using specialized models
- Configurable result count and score thresholds
- Support for both sync and async processing
- Automatic relevance scoring for retrieved content
- Integration with various reranking model providers

## Usage Patterns

### Basic Reranking Pipeline
```python
from datapizza.modules.rerankers.cohere import CohereReranker

reranker = CohereReranker(
    api_key="your-cohere-key",
    endpoint="https://api.cohere.ai/v1",
    top_n=5,
    threshold=0.6
)

query = "What is deep learning?"
reranked_chunks = reranker(query, chunks)
```

### RAG Pipeline Integration
```python
from datapizza.modules.rerankers.together import TogetherReranker
from datapizza.vectorstores import QdrantVectorStore

# Initial broad retrieval
vectorstore = QdrantVectorStore(collection_name="documents")
initial_results = vectorstore.similarity_search(query, k=20)

# Rerank for better relevance
reranker = TogetherReranker(api_key="together-key", model="rerank-model")
reranked_results = reranker(query, initial_results)
```

## Best Practices

1. **Choose the Right Model**: Select reranker models based on your domain and language requirements
2. **Tune Thresholds**: Experiment with relevance score thresholds to balance precision and recall
3. **Initial Retrieval Size**: Retrieve more documents initially (k=20-50) before reranking to improve final quality
4. **Performance Considerations**: Use async processing for high-throughput applications
5. **Cost Management**: Monitor API usage, especially for high-volume applications
6. **Evaluation**: Test different rerankers on your specific data to find the best performance
