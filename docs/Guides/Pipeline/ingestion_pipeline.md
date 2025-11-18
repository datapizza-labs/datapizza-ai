# Ingestion Pipeline


The `IngestionPipeline` provides a streamlined way to process documents, transform them into nodes (chunks of text with metadata), generate embeddings, and optionally store them in a vector database. It allows chaining various components like parsers, captioners, splitters, and embedders to create a customizable document processing workflow.


## Core Concepts

-   **Components**: These are the processing steps in the pipeline, typically inheriting from `datapizza.core.models.PipelineComponent`. Each component implements a `_run` method to perform a specific task like parsing a document, splitting text, or generating embeddings. Components are executed sequentially via their `__call__` method in the order they are provided.
-   **Vector Store**: An optional component responsible for storing the final nodes and their embeddings.
-   **Nodes**: The fundamental unit of data passed between components. A node usually represents a chunk of text (e.g., a paragraph, a table summary) along with its associated metadata and embeddings.

## Available Components

The pipeline typically supports components for:

1.  [**Parsers**](../../API%20Reference/Modules/Parsers/index.md): Convert raw documents (PDF, DOCX, etc.) into structured `Node` objects (e.g., `AzureParser`, `UnstructuredParser`).
2.  [**Captioners**](../../API%20Reference/Modules/captioners.md): Enhance nodes representing images or tables with textual descriptions using models like LLMs (e.g., `LLMCaptioner`).
3.  [**Splitters**](../../API%20Reference/Modules/Splitters/index.md): Divide nodes into smaller chunks based on their content (e.g., `NodeSplitter`, `PdfImageSplitter`).
4.  [**Embedders**](../../API%20Reference/Embedders/openai_embedder.md): Create chunk embeddings for semantic search and similarity matching (e.g., `NodeEmbedder`, `ClientEmbedder`).
     - [`ChunkEmbedder`](../../API%20Reference/Embedders/chunk_embedder.md): Batch processing for efficient embedding of multiple nodes.
5.  [**Vector Stores**](../../API%20Reference/Vectorstore/qdrant_vectorstore.md): Store and retrieve embeddings efficiently using vector databases (e.g., `QdrantVectorstore`).

Refer to the specific documentation for each component type (e.g., in `datapizza.parsers`, `datapizza.embedders`) for details on their specific parameters and usage. Remember that pipeline components typically inherit from `PipelineComponent` and implement the `_run` method.


## Configuration Methods

There are two main ways to configure and use the `IngestionPipeline`:

### 1. Programmatic Configuration

Define and configure the pipeline directly within your Python code. This offers maximum flexibility.

```python
from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

vector_store = QdrantVectorstore(
    location=":memory:" # or set host and port
)
vector_store.create_collection(collection_name="datapizza", vector_config=[VectorConfig(dimensions=1536, name="vector_name")])

pipeline = IngestionPipeline(
    modules=[
        DoclingParser(),
        NodeSplitter(max_char=2000),
        ChunkEmbedder(client=OpenAIClient(api_key="OPENAI_API_KEY", model="text-embedding-3-small"), model_name="text-embedding-3-small", embedding_name="small"),
    ],
    vector_store=vector_store,
    collection_name="datapizza",
)

pipeline.run(file_path="sample.pdf")

print(vector_store.search(query_vector= [0.0]*1536, collection_name="datapizza", k=4))
```


### 2. YAML Configuration

Define the entire pipeline structure, components, and their parameters in a YAML file. This is useful for managing configurations separately from code.

```python
from datapizza.pipeline.pipeline import IngestionPipeline
import os

# Load pipeline from YAML
pipeline = IngestionPipeline().from_yaml("path/to/your/config.yaml")

# Run the pipeline (Ensure necessary ENV VARS for the YAML config are set)
pipeline.run(file_path="path/to/your/document.pdf")
```

#### Example YAML Configuration (`config.yaml`)

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

**Key points for YAML configuration:**

-   **Environment Variables**: Use `${VAR_NAME}` syntax within strings to securely load secrets or configuration from environment variables. Ensure these variables are set in your execution environment.
-   **Clients**: Define shared clients (like `OpenAIClient`) under the `clients` key and reference them by name within module `params`.
-   **Modules**: List components under `modules`. Each requires `type` (class name) and `module` (Python path to the class). `params` are passed to the component's constructor (`__init__`). Components should generally inherit from `PipelineComponent`.
-   **Vector Store**: Configure the optional vector store similarly to modules.
-   **Collection Name**: Must be provided if a `vector_store` is configured.


## Pipeline Execution (`run` method)
```python
pipeline.run(file_path=f, metadata={"name": f, "type": "md"})
```
### Async Execution (`a_run` method)

IngestionPipeline support async run
*NB:* Every modules should implement `_a_run` method to run the async pipeline.

```python
await pipeline.a_run(file_path=f, metadata={"name": f, "type": "md"})
```
