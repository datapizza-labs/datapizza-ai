# Build a RAG

This guide demonstrates how to build a complete RAG (Retrieval-Augmented Generation) system using datapizza-ai's pipeline architecture. We'll cover both the **ingestion pipeline** for processing and storing documents, and the **DagPipeline** for retrieval and response generation.

## Overview

A RAG system consists of two main phases:

1. **Ingestion**: Process documents, split them into chunks, generate embeddings, and store in a vector database
2. **Retrieval**: Query the vector database, retrieve relevant chunks, and generate responses

datapizza-ai provides specialized pipeline components for each phase:

- **IngestionPipeline**: Sequential processing for document ingestion
- **DagPipeline**: Graph-based processing for complex retrieval workflows

## Part 1: Document Ingestion Pipeline

The ingestion pipeline processes raw documents and stores them in a vector database. Here's a complete example:

### Basic Ingestion Setup

```sh
pip install datapizza-ai-parsers-docling
```

```python
from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.captioners import LLMCaptioner
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

vectorstore = QdrantVectorstore(location=":memory:")
vectorstore.create_collection(
    "my_documents",
    vector_config=[VectorConfig(name="text-embedding-3-small", dimensions=1536)]
)

embedder_client = OpenAIEmbedder(
    api_key="YOUR_API_KEY",
    model_name="text-embedding-3-small",
)

ingestion_pipeline = IngestionPipeline(
    modules=[
        DoclingParser(), # choose between Docling, Azure or TextParser to parse plain text

        #LLMCaptioner(
        #    client=OpenAIClient(api_key="YOUR_API_KEY"),
        #), # This is optional, add it if you want to caption the media

        NodeSplitter(max_char=1000),             # Split Nodes into Chunks
        ChunkEmbedder(client=embedder_client),   # Add embeddings to Chunks
    ],
    vector_store=vectorstore,
    collection_name="my_documents"
)

ingestion_pipeline.run("sample.pdf", metadata={"source": "user_upload"})

res = vectorstore.search(
    query_vector = [0.0] * 1536,
    collection_name="my_documents",
    k=2,
)
print(res)
```


### Configuration-Based Ingestion

You can also define your pipeline using YAML configuration:

```yaml
constants:
  EMBEDDING_MODEL: "text-embedding-3-small"
  CHUNK_SIZE: 1000

ingestion_pipeline:
  clients:
    openai_embedder:
      provider: openai
      model: "${EMBEDDING_MODEL}"
      api_key: "${OPENAI_API_KEY}"

  modules:
    - name: parser
      type: DoclingParser
      module: datapizza.modules.parsers.docling
    - name: splitter
      type: NodeSplitter
      module: datapizza.modules.splitters
      params:
        max_char: ${CHUNK_SIZE}
    - name: embedder
      type: ChunkEmbedder
      module: datapizza.embedders
      params:
        client: openai_embedder

  vector_store:
    type: QdrantVectorstore
    module: datapizza.vectorstores.qdrant
    params:
      host: "localhost"
      port: 6333

  collection_name: "my_documents"
```

Load and use the configuration:

```python
from datapizza.pipeline import IngestionPipeline

# Make sure the collection exists before running the pipeline
pipeline = IngestionPipeline().from_yaml("ingestion_pipeline.yaml")
pipeline.run("sample.pdf")

```

## Part 2: Retrieval with DagPipeline

The DagPipeline enables complex retrieval workflows with query rewriting, embedding, and response generation.

### Basic Retrieval Setup

```python
from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

openai_client = OpenAIClient(
    model="gpt-4o-mini",
    api_key="YOUR_API_KEY"
)

query_rewriter = ToolRewriter(
    client=openai_client,
    system_prompt="Rewrite user queries to improve retrieval accuracy."
)

embedder = OpenAIEmbedder(
    api_key="YOUR_API_KEY",
    model_name="text-embedding-3-small"
)

# Use the same qdrant of ingestion (prefer host and port instead of location when possible)
retriever = QdrantVectorstore(location=":memory:")
retriever.create_collection(
    "my_documents",
    vector_config=[VectorConfig(name="embedding", dimensions=1536)]
)

prompt_template = ChatPromptTemplate(
    user_prompt_template="User question: {{user_prompt}}\n:",
    retrieval_prompt_template="Retrieved content:\n{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}"
)

dag_pipeline = DagPipeline()
dag_pipeline.add_module("rewriter", query_rewriter)
dag_pipeline.add_module("embedder", embedder)
dag_pipeline.add_module("retriever", retriever)
dag_pipeline.add_module("prompt", prompt_template)
dag_pipeline.add_module("generator", openai_client)

dag_pipeline.connect("rewriter", "embedder", target_key="text")
dag_pipeline.connect("embedder", "retriever", target_key="query_vector")
dag_pipeline.connect("retriever", "prompt", target_key="chunks")
dag_pipeline.connect("prompt", "generator", target_key="memory")

query = "tell me something about this document"
result = dag_pipeline.run({
    "rewriter": {"user_prompt": query},
    "prompt": {"user_prompt": query},
    "retriever": {"collection_name": "my_documents", "k": 3},
    "generator":{"input": query}
})

print(f"Generated response: {result['generator']}")
```
