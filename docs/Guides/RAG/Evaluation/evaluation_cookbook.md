# Practical Guide with TechCrunch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kCH7d3q95uUk5q64KAJa8Bj4c6WUiRVu?usp=sharing){:target="_blank"}

This cookbook provides a step-by-step walkthrough of evaluating a Retrieval Augmented Generation (RAG) pipeline using a practical example based on the TechCrunch dataset. It demonstrates how to implement various evaluation metrics and techniques discussed in the [RAG Evaluation Overview](./evaluation_overview.md).
A full version of the entire code presented in this cookbook can be downloaded as a Jupyter notebook from [this link](https://git.datapizza.tech/lab-ai/datapizza/researchandevelopment/rag-evaluation/-/blob/develop/notebooks/cookbook_tech_crunch_evaluation.ipynb?ref_type=heads).

The goal is to bridge the gap between theoretical evaluation concepts and their concrete application in a Python environment. We will cover data loading, ingestion, retrieval, answer generation, and a comprehensive suite of evaluation metrics.

## 0. Prerequisites and Setup

Before diving into the code, ensure you have the necessary environment and dependencies set up. The script relies on several libraries, including `datapizza-ai` (having `qdrant-client`, `openai`, `google-generativeai`, `tiktoken`, `cohere`), `tqdm`, `numpy`, `pandas`, `matplotlib`, and `seaborn`.

Environment variables for API keys (OpenAI, Google, Qdrant, Cohere) must be configured, typically in a `.env` file. Ask your DevOps to get these credentials!

```python
# Ensure you have a .env file in your project root or parent directories with:
# OPENAI_API_KEY="your_openai_api_key"
# GOOGLE_API_KEY="your_google_api_key"
# QDRANT_URL="your_qdrant_url_or_localhost"
# QDRANT_API_KEY="your_qdrant_api_key_if_any"
# AZURE_COHERE_API_KEY="your_cohere_api_key"
# AZURE_COHERE_ENDPOINT="your_cohere_endpoint"
```

## 1. Load Dataset

The first step is to load our dataset. This typically consists of two main parts:
1.  **Corpus**: The knowledge base from which the RAG system will retrieve information. In this example, it's a JSON file containing TechCrunch articles.
2.  **Golden Dataset**: A set of questions (queries) with their corresponding ground truth answers and/or evidence documents. This dataset is crucial for evaluation, as explained in the "Building a Golden Dataset" section of the [RAG Evaluation Overview](./evaluation_overview.md).

The dataset can be downloaded [here](https://git.datapizza.tech/lab-ai/datapizza/researchandevelopment/rag-evaluation/-/tree/develop/dataset/MultiHop?ref_type=heads).
Within the downloaded `MultiHop` directory, you'll find several JSON files:

*   `corpus.json` contains the original TechCrunch articles that serve as the knowledge base.
*   `corpus_techcrunch_300.json` is a smaller, sampled version of `corpus.json` (300 articles) used in this cookbook for quicker processing.
*   `MultiHopRAG.json` is the "golden dataset" containing questions, ground truth answers, and evidence, formatted and preprocessed for RAG evaluation.
*   `MultiHopRAG_TechCrunch_300.json` is the sampled version of the golden dataset, corresponding to the 300 articles in `corpus_techcrunch_300.json`.
*   `MultiHopRAG_TechCrunch_300_results.json` is an example file showing the kind of output you can expect after running the retrieval and generation steps described in this cookbook on the sampled dataset.

```python
import json

CORPUS_PATH = "./dataset/MultiHop/corpus_techcrunch_300.json"
GOLDEN_DATASET_PATH = "./dataset/MultiHop/MultiHopRAG_TechCrunch_300.json"

# load the corpus (knowledge base)
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus_data = json.load(f)

# load the golden dataset (queries)
with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
    golden_dataset = json.load(f)
```
The `corpus_data` will be ingested into th Qdrant vector store, while `golden_dataset` will be used to evaluate the RAG pipeline's performance.

## 2. Ingest Dataset

Once the corpus is loaded, it needs to be processed and ingested into a vector database. This involves several steps: document parsing, splitting, embedding, and indexing (i.e. uploading to Qdrant).

```python
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from datapizza.clients.google import GoogleClient
from datapizza.clients.openai import OpenAIClient
from datapizza.embedders import NodeEmbedder
from datapizza.pipeline.pipeline import IngestionPipeline
from datapizza.splitters import RecursiveSplitter
from datapizza.vectorstores import QdrantVectorstore
from datapizza.vectorstores.vectorstore import VectorConfig
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
```

### 2.1. Custom `LLMTreeBuilderRouting`

This cookbook uses a custom `LLMTreeBuilderRouting` class. This class is designed to choose different language models for document summarization/structuring (tree building) based on the document's token count. Shorter documents might use a faster/cheaper model, while longer documents might benefit from a more powerful model.

```python
import tiktoken
from datapizza.clients import Client
from datapizza.treebuilder import LLMTreeBuilder


class LLMTreeBuilderRouting:
    def __init__(
        self,
        client_for_short_documents: Client,
        client_for_long_documents: Client,
        token_threshold: int = 10000,
        tokenizer_model: str = "gpt-4o-mini",
    ):
        if not isinstance(client_for_short_documents, Client):
            raise ValueError(
                "client_for_short_documents must be an instance of datapizza.clients.client.Client"
            )
        if not isinstance(client_for_long_documents, Client):
            raise ValueError(
                "client_for_long_documents must be an instance of datapizza.clients.client.Client"
            )

        self.client_for_short_documents = client_for_short_documents
        self.client_for_long_documents = client_for_long_documents
        self.llm_treebuilder_for_short_documents = LLMTreeBuilder(
            self.client_for_short_documents
        )
        self.llm_treebuilder_for_long_documents = LLMTreeBuilder(
            self.client_for_long_documents
        )
        self.token_threshold = token_threshold
        self.tokenizer_model = tokenizer_model
        self.encoding = tiktoken.encoding_for_model(self.tokenizer_model)

    def run(self, path_to_file: str):
        with open(path_to_file, "r") as f:
            text = f.read()

        num_tokens = len(self.encoding.encode(text))

        if num_tokens > self.token_threshold:
            return self.llm_treebuilder_for_long_documents(path_to_file)
        else:
            return self.llm_treebuilder_for_short_documents(path_to_file)

    def __call__(self, path_to_file: str):
        return self.run(path_to_file)
```
This routing mechanism demonstrates a practical approach to optimizing cost and performance in an ingestion pipeline, using a custom module.

### 2.2. Configuration and Component Initialization

This section loads environment variables and defines key configuration parameters for the ingestion pipeline. The `initialize_components` function sets up clients (OpenAI, Google), the `LLMTreeBuilderRouting`, splitter, embedder, and vector store.

```python
# Load environment variables from .env file
load_dotenv()
# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_LLM_MODEL = "gpt-4o-mini"
EMBEDDING_NAME = "openai_large"
EMBEDDING_DIMENSIONS = 3072

QDRANT_HOST = os.getenv("QDRANT_URL", "localhost")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional
COLLECTION_NAME = "test-evaluation-tech-crunch-collection"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_LLM_MODEL = "gemini-2.5-pro-preview-03-25"

SPLITTER_MAX_CHAR = 3000
SPLITTER_OVERLAP = 200

WORKING_FOLDER_PATH = (
    "./data/single_corpus_techcrunch_300"
)


def initialize_components():
    """Initializes and configures all necessary components."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    logging.info("Initializing OpenAI clients...")
    embed_client = OpenAIClient(api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL)
    client_for_short_documents = OpenAIClient(
        api_key=OPENAI_API_KEY, model=OPENAI_LLM_MODEL
    )
    client_for_long_documents = GoogleClient(
        api_key=GOOGLE_API_KEY, model=GOOGLE_LLM_MODEL
    )

    logging.info("Initializing TreeBuilder...")
    treebuilder = LLMTreeBuilderRouting(
        client_for_short_documents=client_for_short_documents,
        client_for_long_documents=client_for_long_documents,
    )

    logging.info("Initializing Splitter...")
    splitter = RecursiveSplitter(max_char=SPLITTER_MAX_CHAR, overlap=SPLITTER_OVERLAP)

    logging.info("Initializing Embedder...")
    embedder = NodeEmbedder(
        client=embed_client,
        model_name=OPENAI_EMBEDDING_MODEL,
        embedding_name=EMBEDDING_NAME,
    )

    logging.info("Initializing Vectorstore...")
    vectorstore = QdrantVectorstore(
        host=QDRANT_HOST,
        port=None,
        api_key=QDRANT_API_KEY,
    )

    os.makedirs(WORKING_FOLDER_PATH, exist_ok=True)
    return treebuilder, splitter, embedder, vectorstore
```

### 2.3. Vector Store Collection Creation

Before ingesting data, we ensure the target collection exists in Qdrant. The `VectorConfig` specifies the embedding model's name and dimensions.

```python
def create_collection_if_not_exists(
    vectorstore: QdrantVectorstore, collection_name: str
):
    """Creates the collection if it doesn't exist."""
    logging.info(f"Creating collection '{collection_name}'...")
    vectorstore.create_collection(
        collection_name=collection_name,
        vector_config=[
            VectorConfig(
                name=EMBEDDING_NAME,
                dimensions=EMBEDDING_DIMENSIONS,
            ),
        ],
    )
    logging.info(f"Collection '{collection_name}' created successfully.")
```

### 2.4. Document Processing

The `process_single_document` function handles the ingestion of individual documents. It saves the document's body to a temporary file* and then runs the ingestion pipeline on it. The `process_documents` function orchestrates this for all documents in the corpus, using `ThreadPoolExecutor` for parallel processing to speed up ingestion.

\*The temporary files are not deleted in this code, remember to clean if you want.
```python
def process_single_document(
    i: int, doc: dict, pipeline: IngestionPipeline, folder_path: str
):
    """Processes a single document: saves it and runs the pipeline."""
    title = doc.get("title")
    source = doc.get("source")
    url = doc.get("url")
    date = doc.get("date")
    author = doc.get("author")
    published_at = doc.get("published_at")
    category = doc.get("category")

    text = doc.get("body")
    if not text:
        logging.warning(
            f"Skipping document {i + 1} ('{title}') due to missing 'body' field."
        )
        return title, False

    file_path = os.path.join(folder_path, f"document_{i}.txt")
    logging.info(f"Processing document {i + 1}: '{title}' -> {file_path}")
    try:
        with open(file_path, "w") as f:
            f.write(text)
        pipeline.run(
            file_path=file_path,
            metadata={
                "document_name": title,
                "file_path": file_path,
                "source": source,
                "url": url,
                "date": date,
                "author": author,
                "published_at": published_at,
                "category": category,
            },
        )
        return title, True
    except Exception as e:
        logging.error(f"Error processing document {i + 1} ('{title}'): {e}")
        return title, False


def process_documents(
    pipeline: IngestionPipeline, json_path: str, folder_path: str, max_workers: int = 10
):
    """Loads documents from a JSON file and runs them through the ingestion pipeline in parallel."""
    logging.info(f"Loading documents from {json_path}...")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: Input JSON file not found at {json_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {json_path}")
        return

    num_docs = len(data)
    logging.info(
        f"Found {num_docs} documents. Starting parallel ingestion with {max_workers} workers..."
    )

    futures = []
    processed_count = 0
    failed_count = 0
    skipped_count = 0

    os.makedirs(folder_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, doc in enumerate(data):
            futures.append(
                executor.submit(process_single_document, i, doc, pipeline, folder_path)
            )

        for future in as_completed(futures):
            try:
                title, success = future.result()
                if success is None:
                    skipped_count += 1
                elif success:
                    processed_count += 1
                    logging.info(
                        f"Successfully processed '{title}' ({processed_count}/{num_docs})"
                    )
                else:
                    original_index = -1
                    for idx, submitted_future in enumerate(futures):
                        if submitted_future == future:
                            original_index = idx
                            break
                    if original_index != -1 and not data[original_index].get("body"):
                        skipped_count += 1
                    else:
                        failed_count += 1
                        logging.warning(f"Failed or skipped processing for '{title}'")

            except Exception as exc:
                failed_count += 1
                logging.error(f"An error occurred during processing: {exc}")

    logging.info("Ingestion process completed.")
    logging.info(
        f"Summary: Processed={processed_count}, Failed={failed_count}, Skipped={skipped_count} out of {num_docs} documents."
    )
```

### 2.5. Running the Ingestion Pipeline

This block initializes the components, creates the Qdrant collection if it doesn't exist, configures the `IngestionPipeline` with the treebuilder, splitter, and embedder modules, and then starts the document processing.

```python
treebuilder, splitter, embedder, vectorstore = initialize_components()

create_collection_if_not_exists(vectorstore, COLLECTION_NAME)

logging.info("Configuring ingestion pipeline...")
pipeline = IngestionPipeline(
    modules=[
        treebuilder,
        splitter,
        embedder,
    ],
    vector_store=vectorstore,
    collection_name=COLLECTION_NAME,
)

process_documents(pipeline, CORPUS_PATH, WORKING_FOLDER_PATH)
```

## 3. Retrieve and Generate Answers

After ingesting the corpus, the next phase is to build a RAG pipeline that can retrieve relevant chunks for a given query and then generate an answer based on these chunks.

```python
import tqdm
from datapizza.core.models import PipelineComponent
from datapizza.embedders import ClientEmbedder
from datapizza.pipeline.retrieval_pipeline import DagPipeline
from datapizza.prompt import ChatPromptTemplate
from datapizza.rerankers.cohere_reranker import CohereReranker
from datapizza.type import Chunk
```

### 3.1. Custom `ChunkTransformer`

This custom pipeline component, `ChunkTransformer`, is used to reformat retrieved chunks before they are passed to the reranker and the generator. It constructs a string with the title, URL, and content of each chunk. This can help provide better context to downstream components.

```python
class ChunkTransformer(PipelineComponent):
    def __init__(self):
        pass

    def __call__(self, chunks: list[Chunk]):
        return [self.process_chunk(chunk) for chunk in chunks]

    def process_chunk(self, chunk: Chunk):
        stringified_chunk = f"""Title: {chunk.metadata.get("document_name")}
Url: {chunk.metadata.get("url")}
Content: {chunk.text}"""
        return Chunk(
            id=chunk.id,
            text=stringified_chunk,
            metadata=chunk.metadata,
        )
```

### 3.2. Pipeline Components and Construction

The `create_components` function initializes all modules needed for the retrieval and generation pipeline:

*   `ClientEmbedder`: For embedding the input query.
*   `QdrantVectorstore`: For retrieving chunks from the indexed corpus.
*   `ChunkTransformer`: The custom component defined above.
*   `CohereReranker`: For reranking the retrieved chunks to improve relevance.
*   `ChatPromptTemplate`: For formatting the input to the generator LLM, including the retrieved context and user query.
*   `OpenAIClient` (as generator): The LLM that generates the final answer.

The `build_pipeline` function then assembles these components into a `DagPipeline`, defining the flow of data between modules.

```python
AZURE_COHERE_API_KEY = os.getenv("AZURE_COHERE_API_KEY")
AZURE_COHERE_ENDPOINT = os.getenv("AZURE_COHERE_ENDPOINT")

def create_components():
    client_embedder = OpenAIClient(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
    embedder = ClientEmbedder(
        client=client_embedder,
        model_name=OPENAI_EMBEDDING_MODEL,
        embedding_name=EMBEDDING_NAME,
    )
    retriever = QdrantVectorstore(
        host=QDRANT_HOST,
        port=None,
        api_key=QDRANT_API_KEY,
    )
    chunk_transfomer = ChunkTransformer()
    reranker = CohereReranker(
        api_key=AZURE_COHERE_API_KEY,
        endpoint=AZURE_COHERE_ENDPOINT,
        top_n=10,
        threshold=0.1,
    )
    retrieval_prompt_template = """
Here are the context documents:
{% for chunk in chunks %}
{{chunk.text}}
{% endfor %}
"""
    template = ChatPromptTemplate(
        user_prompt_template="{{user_prompt}}",
        system_prompt_template="You are a helpful assistant. Try to answer user questions given the context",
        retrieval_prompt_template=retrieval_prompt_template,
    )
    generator = OpenAIClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return {
        "embedder": embedder,
        "retriever": retriever,
        "chunk_transformer": chunk_transfomer,
        "reranker": reranker,
        "prompt_template": template,
        "generator": generator,
    }


def build_pipeline():
    pipeline = DagPipeline()
    components = create_components()

    pipeline.add_module("embedder", components["embedder"])
    pipeline.add_module("retriever", components["retriever"])
    pipeline.add_module("reranker", components["reranker"])
    pipeline.add_module("prompt_template", components["prompt_template"])
    pipeline.add_module("generator", components["generator"])
    pipeline.add_module("chunk_transformer", components["chunk_transformer"])

    pipeline.connect("embedder", "retriever", target_key="query_vector")
    pipeline.connect("retriever", "chunk_transformer", target_key="chunks")
    pipeline.connect("chunk_transformer", "reranker", target_key="documents")
    pipeline.connect("reranker", "prompt_template", target_key="chunks")
    pipeline.connect("prompt_template", "generator", target_key="memory")

    return pipeline
```

### 3.3. Generating Results

The `generate_results` function takes a data item (containing a query and ground truth answer) and the RAG pipeline, then runs the pipeline to get a predicted answer. The `process_data_item` function is a wrapper that builds a pipeline instance (important for parallel processing where each process needs its own pipeline instance) and calls `generate_results`. It formats the output including the query, ground truth answer, predicted answer, retrieved chunks, and ground truth chunks.

The main script then uses `ThreadPoolExecutor` to process all items in the `golden_dataset` in parallel, collecting the results.

```python
def save_results(path: str, results: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f)

def generate_results(data: dict, pipeline: DagPipeline):
    user_input = data["query"]

    results = pipeline.run(
        {
            "embedder": {"text": user_input},
            "retriever": {"collection_name": COLLECTION_NAME, "k": 10},
            "reranker": {"query": user_input},
            "prompt_template": {
                "user_prompt": user_input,
                "retrieval_query": user_input,
            },
            "generator": {
                "input": None,
                "system_prompt": "You are a helpful assistant. Try to answer user questions given the context. Use as few words as possible.",
            },
        }
    )

    logging.info(user_input)
    logging.info("--------------------------------")
    logging.info(f"GT: {data['answer']}")
    logging.info("--------------------------------")
    logging.info(f"Pred: {results['generator'].text}")
    logging.info("--------------------------------")
    return results

def process_data_item(data_item: dict):
    local_pipeline = build_pipeline()

    try:
        results = generate_results(data_item, local_pipeline)
    except Exception as e:
        logging.error(f"Error processing item {data_item.get('query', 'UNKNOWN')}: {e}")
        return {
            "query": data_item.get("query"),
            "answer": data_item.get("answer"),
            "prediction": f"Error: {e}",
            "chunks": [],
            "gt_chunks": [
                {"title": x.get("title"), "fact": x.get("fact")}
                for x in data_item.get("evidence_list", [])
            ],
            "error": str(e),
        }

    prediction = results.get("generator")
    prediction_text = (
        prediction.text if prediction else "Error: Generator result missing"
    )

    reranker_results = results.get("reranker", [])

    return {
        "query": data_item["query"],
        "answer": data_item["answer"],
        "prediction": prediction_text,
        "chunks": [
            {
                "id": chunk.id,
                "title": chunk.metadata.get("document_name", "N/A"),
                "text": chunk.metadata.get("text", "N/A"), # This should be chunk.text after transformer
            }
            for chunk in reranker_results # Assuming reranker returns Chunk objects with metadata
        ],
        "gt_chunks": [
            {"title": x["title"], "fact": x["fact"]} for x in data_item["evidence_list"]
        ],
    }

# Main execution block for generating results
results_list = []
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [
        executor.submit(process_data_item, golden_dataset[i])
        for i in range(len(golden_dataset))
    ]
    for future in tqdm.tqdm(
        as_completed(futures), total=len(futures), desc="Processing queries"
    ):
        results_list.append(future.result())

save_results(
    path="./data/results/MultiHopRAG_TechCrunch_300_results.json",
    results=results_list,
)
```
For clarity in the `process_data_item` returned chunks:
The `ChunkTransformer` sets `chunk.text` to the stringified version.
If `reranker_results` are these transformed `Chunk` objects, then `chunk.text` will be the transformed string.
The metadata like `document_name` is preserved.
The original text before transformation is in `chunk.metadata.get("text")` which is used in the eval.

## 4. Evaluation using `datapizza-ai` metrics

This is the core of the evaluation process, where we use the generated results and the golden dataset to calculate various metrics. These metrics help quantify the performance of both the retrieval and generation components of the RAG pipeline. This section directly implements concepts from the [RAG Evaluation Overview](./evaluation_overview.md).

```python
with open("./data/results/MultiHopRAG_TechCrunch_300_results.json", "r") as f:
    results_list = json.load(f)

embed_client = OpenAIClient(api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL)
vectorstore = QdrantVectorstore(
    host=QDRANT_HOST,
    port=None,
    api_key=QDRANT_API_KEY,
)

from typing import Optional
import numpy as np
import pandas as pd
from datapizza.clients.google_client import GoogleClient
from datapizza.evaluation import metrics
from pydantic import BaseModel
```

### 4.1. Defining "Relevance": Preparing Embeddings for Similarity Metrics

As discussed in the [RAG Evaluation Overview](./evaluation_overview.md) under "Defining Relevance: Exact Match vs. Cosine Similarity", similarity-based metrics require embeddings for both retrieved and ground truth chunks.

The `get_embeddings_retrieved_chunks` function fetches embeddings for the text of retrieved chunks. It does this by dumping all chunks and their vectors from the Qdrant collection and creating a mapping from text to embedding. This is an expensive operation and can be optimized if chunk IDs are consistently used and retrievable.
The `get_embeddings_ground_truth_chunks` function generates embeddings for the ground truth texts on the fly using the `embed_client`.

```python
def get_embeddings_retrieved_chunks(
    retrieved_chunk_texts: list[str],
    vector_store: QdrantVectorstore,  # Expects an instance of QdrantVectorStore (or compatible)
    collection_name: str,
) -> list[Optional[list[float]]]:
    """
    Retrieves embeddings for a list of text chunks from a Qdrant collection.

    This function dumps all chunks with their vectors from the specified collection,
    maps their texts to their embeddings, and then looks up the embeddings for
    the provided list of chunk texts.

    Args:
        retrieved_chunk_texts: A list of strings, where each string is the text
                               of a chunk whose embedding is to be retrieved.
        vector_store: An instance of a vector store client that has a
                      `dump_collection` method (e.g., QdrantVectorStore).
                      The `dump_collection` method should yield chunk-like objects
                      that have `payload` (a dictionary) and `vector` attributes.
        collection_name: The name of the collection in the vector store.
        payload_text_key: The key in the chunk's payload dictionary that
                          stores the text content. Defaults to "text".

    Returns:
        A list of embeddings corresponding to the retrieved_chunk_texts.
        Each embedding is expected to be a list of floats.
        If a text from retrieved_chunk_texts is not found in the collection
        or if its corresponding chunk has no vector, None is placed at that
        position in the returned list.
    """

    all_dumped_chunks = vector_store.dump_collection(
        collection_name=collection_name,
        with_vectors=True,  # Ensure vectors (embeddings) are fetched
    )

    text_to_embedding_map: dict[str, list[float]] = {}
    for chunk in all_dumped_chunks:
        if (
            hasattr(chunk, "text")
            and hasattr(chunk, "embeddings")
            and chunk.embeddings is not None
        ):
            if chunk.text is not None:
                text_to_embedding_map[chunk.text] = chunk.embeddings[0].vector

    retrieved_embeddings: list[Optional[list[float]]] = []
    for text_to_find in retrieved_chunk_texts:
        embedding = text_to_embedding_map.get(text_to_find)
        retrieved_embeddings.append(embedding)

    return retrieved_embeddings

def get_embeddings_ground_truth_chunks(ground_truth_texts, client_embedder):
    embeddings = client_embedder.embed(ground_truth_texts)
    return embeddings
```


### 4.2. Retrieval Evaluation Metrics

These metrics assess how well the retriever found relevant information.

#### 4.2.1. Exact Match Metrics
These metrics consider a retrieved chunk relevant if its text exactly matches a ground truth chunk text.
Refer to [Precision@k, Recall@k, F1-score@k, Hybrid Log-Rank Score (Exact Match)](./evaluation_overview.md#precisionk) in the overview.

```python
def calculate_exact_match_metrics(retrieved_chunk_texts, ground_truth_texts, k_values):
    metrics_dict = {}
    for k_val in k_values:
        metrics_dict[f"precision_at_{k_val}_exact"] = metrics.precision_at_k_exact(
            retrieved_chunks=retrieved_chunk_texts,
            ground_truth_chunks=ground_truth_texts,
            k=k_val,
        )
        metrics_dict[f"recall_at_{k_val}_exact"] = metrics.recall_at_k_exact(
            retrieved_chunks=retrieved_chunk_texts,
            ground_truth_chunks=ground_truth_texts,
            k=k_val,
        )
        metrics_dict[f"f1_at_{k_val}_exact"] = metrics.f1_at_k_exact(
            retrieved_chunks=retrieved_chunk_texts,
            ground_truth_chunks=ground_truth_texts,
            k=k_val,
        )
    metrics_dict["hybrid_log_rank_score_exact"] = metrics.hybrid_log_rank_score_exact(
        retrieved_chunks=retrieved_chunk_texts,
        ground_truth_chunks=ground_truth_texts,
    )
    return metrics_dict
```

#### 4.2.2. Lexical Overlap Metrics (BLEU & ROUGE)
While BLEU and ROUGE are traditionally used for evaluating generated text (like translations or summaries), this script applies `corpus_bleu_score` and `corpus_rouge_scores` to compare the set of retrieved chunk texts against the set of ground truth chunk texts. This can provide a measure of lexical overlap at a corpus level.
Refer to [BLEU Score](./evaluation_overview.md#bleu-score-bilingual-evaluation-understudy) and [ROUGE Score](./evaluation_overview.md#rouge-score-recall-oriented-understudy-for-gisting-evaluation) in the overview for their typical usage.

```python
def calculate_bleu_rouge_metrics(retrieved_chunk_texts, ground_truth_texts):
    metrics_dict = {}
    bleu_value = metrics.corpus_bleu_score(
        retrieved_chunks=retrieved_chunk_texts, ground_truth_chunks=ground_truth_texts
    )
    metrics_dict["corpus_bleu"] = bleu_value

    rouge_values = metrics.corpus_rouge_scores(
        retrieved_chunks=retrieved_chunk_texts, ground_truth_chunks=ground_truth_texts
    )
    for rouge_type, scores_dict in rouge_values.items():
        metrics_dict[f"corpus_{rouge_type}_precision"] = scores_dict["precision"]
        metrics_dict[f"corpus_{rouge_type}_recall"] = scores_dict["recall"]
        metrics_dict[f"corpus_{rouge_type}_fmeasure"] = scores_dict["fmeasure"]
    return metrics_dict
```

#### 4.2.3. Similarity-Based Metrics
These metrics use vector embeddings and cosine similarity to determine relevance, as explained in the [RAG Evaluation Overview](./evaluation_overview.md#defining-relevance-exact-match-vs-cosine-similarity). A retrieved chunk is relevant if its embedding is semantically similar to a ground truth chunk's embedding above a certain `similarity_threshold`.
Refer to [Precision@k, Recall@k, F1-score@k, Hybrid Log-Rank Score (Similarity-based)](./evaluation_overview.md#precisionk) in the overview.

```python
def calculate_similarity_based_metrics(
    retrieved_chunk_embeddings: list[np.ndarray],
    ground_truth_embeddings: list[np.ndarray],
    k_values: list[int],
    similarity_threshold: float = 0.8,
    hybrid_log_rank_gamma: float = 1.0,
    hybrid_log_rank_alpha: float = 0.5,
):
    metrics_dict = {}
    if not retrieved_chunk_embeddings or not ground_truth_embeddings:
        for k_val in k_values:
            metrics_dict[f"precision_at_{k_val}_similarity"] = None
            metrics_dict[f"recall_at_{k_val}_similarity"] = None
            metrics_dict[f"f1_at_{k_val}_similarity"] = None
        metrics_dict["hybrid_log_rank_score_similarity"] = None
        return metrics_dict

    for k_val in k_values:
        try:
            metrics_dict[f"precision_at_{k_val}_similarity"] = (
                metrics.precision_at_k_similarity(
                    retrieved_embeddings=retrieved_chunk_embeddings,
                    ground_truth_embeddings=ground_truth_embeddings,
                    k=k_val,
                    similarity_threshold=similarity_threshold,
                )
            )
        except Exception as e:
            logging.error(f"Error calculating precision_at_{k_val}_similarity: {e}")
            metrics_dict[f"precision_at_{k_val}_similarity"] = None
        # Similar try-except blocks for recall and F1
        try:
            metrics_dict[f"recall_at_{k_val}_similarity"] = (
                metrics.recall_at_k_similarity(
                    retrieved_embeddings=retrieved_chunk_embeddings,
                    ground_truth_embeddings=ground_truth_embeddings,
                    k=k_val,
                    similarity_threshold=similarity_threshold,
                )
            )
        except Exception as e:
            logging.error(f"Error calculating recall_at_{k_val}_similarity: {e}")
            metrics_dict[f"recall_at_{k_val}_similarity"] = None

        try:
            metrics_dict[f"f1_at_{k_val}_similarity"] = metrics.f1_at_k_similarity(
                retrieved_embeddings=retrieved_chunk_embeddings,
                ground_truth_embeddings=ground_truth_embeddings,
                k=k_val,
                similarity_threshold=similarity_threshold,
            )
        except Exception as e:
            logging.error(f"Error calculating f1_at_{k_val}_similarity: {e}")
            metrics_dict[f"f1_at_{k_val}_similarity"] = None

    try:
        metrics_dict["hybrid_log_rank_score_similarity"] = (
            metrics.hybrid_log_rank_score_similarity(
                retrieved_embeddings=retrieved_chunk_embeddings,
                ground_truth_embeddings=ground_truth_embeddings,
                similarity_threshold=similarity_threshold,
                gamma=hybrid_log_rank_gamma,
                alpha=hybrid_log_rank_alpha,
            )
        )
    except Exception as e:
        logging.error(f"Error calculating hybrid_log_rank_score_similarity: {e}")
        metrics_dict["hybrid_log_rank_score_similarity"] = None
    return metrics_dict
```

### 4.3. Generation Evaluation Metrics

These metrics assess the quality of the answer generated by the LLM.

#### 4.3.1. LLM-as-Judge
This approach uses another powerful LLM (the "judge") to evaluate the generated answer's quality against the ground truth answer and the original query. The script defines a `MatchingResult` Pydantic model for the judge's structured output.
Refer to [LLM-as-judge](./evaluation_overview.md#llm-as-judge) in the overview.

```python
def calculate_answer_generation_metrics(
    query, predicted_answer, ground_truth_answer, client_judge
):
    class MatchingResult(BaseModel):
        is_matching: bool
        # reasoning: str | None = None # Optional

    judge_system_prompt = """You are an AI assistant acting as an impartial judge.
    Your task is to determine if the 'PREDICTION' accurately and satisfactorily answers the 'ORIGINAL QUERY', considering the provided 'ANSWER' as a reference.
    Respond with a JSON object containing two keys:
    1.  `is_matching`: a boolean value (true if the prediction matches, false otherwise).
    2.  `reasoning`: a brief explanation for your decision, especially if it's not a match.

    Focus on semantic similarity and factual correctness. Minor phrasing differences are acceptable if the meaning is preserved.
    If the prediction is too vague, incomplete, or factually incorrect compared to the answer and query, it is not matching.
    """

    judge_input_prompt = f"""ORIGINAL QUERY:
{query}

GROUND TRUTH ANSWER:
{ground_truth_answer}

PREDICTION:
{predicted_answer}"""

    try:
        client_response = client_judge.structured_response(
            input=judge_input_prompt,
            output_cls=MatchingResult,
            system_prompt=judge_system_prompt,
        )
        matching_data = client_response.structured_data[0]
        return int(matching_data.is_matching)
    except Exception as e:
        logging.error(f"Error calculating answer generation metrics: {e}")
        return 0 # Default to not matching on error
```

### 4.4. Running the Evaluation Loop

This is the main loop where metrics are calculated for each item in `results_list`.
It first prepares global sets of retrieved and ground truth chunk texts and their embeddings to avoid redundant computations. Then, it iterates through each evaluated query, extracts the necessary data (retrieved chunks, ground truth chunks, predicted answer, etc.), and calls the respective metric calculation functions.

```python
# Main evaluation process
all_item_metrics = []
K_VALUES = [1, 3, 5, 10]

# Prepare embeddings for all unique chunks once
all_retrieved_chunk_texts_set = set()
for item in results_list:
    for chunk_info in item.get("chunks", []):
        # The script stores retrieved chunk texts under 'text' key inside 'chunks' list of dicts.
        # Example: item['chunks'] = [{'id': ..., 'title': ..., 'text': 'actual chunk text from reranker'}]
        if chunk_info.get("text"): # Ensure text is not None or empty
             all_retrieved_chunk_texts_set.add(chunk_info.get("text"))

all_ground_truth_texts_set = set()
for item in results_list:
    for gt_chunk in item.get("gt_chunks", []):
        # Ground truth facts are stored under 'fact' key
        if gt_chunk.get("fact"): # Ensure fact is not None or empty
            all_ground_truth_texts_set.add(gt_chunk.get("fact"))

# Convert sets to lists for functions expecting lists
list_all_retrieved_chunk_texts = list(all_retrieved_chunk_texts_set)
list_all_ground_truth_texts = list(all_ground_truth_texts_set)

# Fetch all retrieved chunks embeddings
# Ensure that the 'text' field used to build list_all_retrieved_chunk_texts is the same text
# that get_embeddings_retrieved_chunks expects to find in the vector store dump.
all_retrieved_chunk_embeddings_list = get_embeddings_retrieved_chunks(
    list_all_retrieved_chunk_texts, vectorstore, COLLECTION_NAME
)
# Compute all ground truth chunks embeddings
all_ground_truth_embeddings_list = get_embeddings_ground_truth_chunks(
    list_all_ground_truth_texts, embed_client
)

# Reorganize into dictionaries for quick lookup: text -> embedding
retrieved_chunks_emb_dict = {
    text: emb
    for text, emb in zip(list_all_retrieved_chunk_texts, all_retrieved_chunk_embeddings_list)
    if emb is not None # Only include if embedding was found
}
ground_truth_emb_dict = {
    text: emb
    for text, emb in zip(list_all_ground_truth_texts, all_ground_truth_embeddings_list)
    if emb is not None
}

client_judge = GoogleClient(
    api_key=GOOGLE_API_KEY,
    model="gemini-2.5-flash-preview-04-17",
)

logging.info("Starting evaluation of each item...")

queries_for_gen_eval = []
ground_truth_answers_for_gen_eval = []
predicted_answers_for_gen_eval = []

for i, item in enumerate(results_list):
    if i % 20 == 0 and i > 0:
        logging.info(f"  Processed {i}/{len(results_list)} items for retrieval metrics...")

    # For retrieval metrics:
    # Extracted from item['chunks'] which are the reranked chunks.
    retrieved_chunk_texts_for_item = [
        chunk.get("text", "") for chunk in item.get("chunks", []) if chunk.get("text")
    ]
    retrieved_chunk_embeddings_for_item = [
        retrieved_chunks_emb_dict.get(text) for text in retrieved_chunk_texts_for_item if retrieved_chunks_emb_dict.get(text) is not None
    ]

    # Ground truth texts (facts) for retrieval
    ground_truth_texts_for_item = []
    if item.get("gt_chunks") and isinstance(item["gt_chunks"], list):
        ground_truth_texts_for_item = [
            gt_chunk.get("fact", "")
            for gt_chunk in item["gt_chunks"]
            if isinstance(gt_chunk, dict) and gt_chunk.get("fact")
        ]
    ground_truth_embeddings_for_item = [
        ground_truth_emb_dict.get(text) for text in ground_truth_texts_for_item if ground_truth_emb_dict.get(text) is not None
    ]

    current_metrics = {"query_id": i, "query": item.get("query", "N/A")}

    exact_metrics = calculate_exact_match_metrics(
        retrieved_chunk_texts_for_item, ground_truth_texts_for_item, K_VALUES
    )
    current_metrics.update(exact_metrics)

    bleu_rouge_metrics = calculate_bleu_rouge_metrics(
        retrieved_chunk_texts, ground_truth_texts
    )
    current_metrics.update(bleu_rouge_metrics)

    similarity_metrics = calculate_similarity_based_metrics(
        retrieved_chunk_embeddings_for_item, ground_truth_embeddings_for_item, K_VALUES,
        similarity_threshold=0.6
    )
    current_metrics.update(similarity_metrics)

    # Store data for batch generation metric calculation
    queries_for_gen_eval.append(item.get("query", ""))
    ground_truth_answers_for_gen_eval.append(item.get("answer", ""))
    predicted_answers_for_gen_eval.append(item.get("prediction", ""))

    all_item_metrics.append(current_metrics)

# Calculate answer generation metrics in parallel
def process_answer_metrics_for_item(query_pred_gt_tuple):
    q, p, g = query_pred_gt_tuple
    return calculate_answer_generation_metrics(q, p, g, client_judge)

logging.info("Calculating answer generation metrics...")
with ThreadPoolExecutor(max_workers=20) as executor: # As per script
    answer_generation_metric_results = list(
        tqdm.tqdm(
            executor.map(
                process_answer_metrics_for_item,
                zip(queries_for_gen_eval, predicted_answers_for_gen_eval, ground_truth_answers_for_gen_eval),
            ),
            total=len(queries_for_gen_eval),
            desc="Evaluating generated answers"
        )
    )

# Add generation metrics to each item's metrics
for i, item_metrics_dict in enumerate(all_item_metrics):
    item_metrics_dict["llm_as_judge_answer_metrics"] = answer_generation_metric_results[i]

logging.info("
Evaluation processing complete.")
# `all_item_metrics` now contains all calculated metrics for each query.
```

### 4.5. Visualizing Results

The `visualize_results` function takes the list of all item metrics and K values to generate and log summary statistics and plots. It uses `pandas` for data manipulation and `matplotlib`/`seaborn` for plotting.

Key visualizations include:

*   A table of summary statistics (mean, median, min, max, std, count, missing_count) for all numeric metrics.
*   A bar chart of average scores for evaluation metrics.
*   Box plots showing the distribution of key metrics (e.g., precision@k, recall@k for different k).

This function is extensive and primarily focuses on presentation. Below is its definition. To see the plots, you would typically run the `plt.show()` commands (commented out in the original script's logging messages) in a Jupyter environment.

```python
import matplotlib.pyplot as plt
import seaborn as sns
# Ensure logging and os are imported if not already:
# import logging
# import os
# import numpy as np # for numeric_cols selection
# import pandas as pd # for DataFrame

def visualize_results(all_item_metrics: list[dict], k_values: list[int]):
    if not all_item_metrics:
        logging.info("🚫 No evaluation data was provided. Skipping visualization.")
        return

    try:
        df_results = pd.DataFrame(all_item_metrics)
    except Exception as e:
        logging.error(f"❌ Error creating DataFrame from all_item_metrics: {e}")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 150)
    pd.set_option("display.float_format", "{:.4f}".format)

    logging.info("
📊======= Overall Evaluation Summary =======📊")
    numeric_cols = df_results.select_dtypes(include=np.number).columns

    if not numeric_cols.empty:
        summary_stats = (
            df_results[numeric_cols]
            .agg(["mean", "median", "min", "max", "std", "count"])
            .transpose()
        )
        summary_stats["missing_count"] = len(df_results) - summary_stats["count"]
        logging.info("
📋 Summary Statistics for Numeric Metrics:")
        try:
            logging.info(f"
{summary_stats.to_string()}")
        except Exception as e:
            logging.error(f"Error logging.infoing summary_stats: {e}")

        plt.style.use("seaborn-v0_8-whitegrid")
        mean_metrics = df_results[numeric_cols].mean(skipna=True).dropna()
        mean_metrics_to_plot = mean_metrics.drop("query_id", errors="ignore")

        if not mean_metrics_to_plot.empty:
            plt.figure(figsize=(15, 8))
            colors = sns.color_palette("viridis", len(mean_metrics_to_plot))
            bars = mean_metrics_to_plot.sort_values(ascending=False).plot(
                kind="bar", color=colors
            )
            plt.title("Average Scores for Evaluation Metrics", fontsize=18, fontweight="bold")
            plt.ylabel("Average Score", fontsize=14)
            plt.xlabel("Metric", fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            for bar in bars.patches:
                bars.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9,
                )
            plt.tight_layout()
            logging.info("
📈 [Plot Code: Average Metrics Bar Chart - Copy and run in a new cell to display `plt.show()`]")
        else:
            logging.info("
⚠️ No numeric metrics with non-NaN mean values to plot.")

        key_metrics_for_boxplot = []
        for k in k_values:
            key_metrics_for_boxplot.extend([f"precision_at_{k}_exact", f"recall_at_{k}_exact", f"f1_at_{k}_exact"])
            key_metrics_for_boxplot.extend([f"precision_at_{k}_similarity", f"recall_at_{k}_similarity", f"f1_at_{k}_similarity"]) # Added similarity

        if "hybrid_log_rank_score_exact" in numeric_cols: key_metrics_for_boxplot.append("hybrid_log_rank_score_exact")
        if "hybrid_log_rank_score_similarity" in numeric_cols: key_metrics_for_boxplot.append("hybrid_log_rank_score_similarity") # Added similarity
        if "llm_as_judge_answer_metrics" in numeric_cols: key_metrics_for_boxplot.append("llm_as_judge_answer_metrics") # Added judge metric

        plot_metrics = [m for m in key_metrics_for_boxplot if m in df_results.columns and pd.api.types.is_numeric_dtype(df_results[m])]
        plot_metrics_valid = [metric for metric in plot_metrics if not df_results[metric].isnull().all()]

        if plot_metrics_valid:
            plt.figure(figsize=(16, max(10, len(plot_metrics_valid) * 0.6)))
            ax_box = plt.gca()
            sns.boxplot(data=df_results[plot_metrics_valid], orient="h", palette="pastel", ax=ax_box)
            ax_box.set_title("Distribution of Key Evaluation Metrics", fontsize=18, fontweight="bold")
            ax_box.set_xlabel("Score", fontsize=14); ax_box.set_ylabel("Metric", fontsize=14)
            ax_box.tick_params(axis='both', labelsize=10); ax_box.grid(axis="x", linestyle="--", alpha=0.7)
            plt.tight_layout()
            logging.info("
📈 [Plot Code: Key Metric Distributions (Box Plots) - Copy and run in a new cell to display `plt.show()`]")
        else:
            logging.info("
⚠️ No key metrics available or valid for box plot distribution.")
    else:
        logging.info("
⚠️ No numeric columns found to generate statistics or plots.")

    logging.info("
📄======= Detailed Results (First 5 Queries) =======📄")
    try:
        logging.info(f"
{df_results.head(5).to_string()}")
    except Exception as e:
        logging.error(f"Error logging.infoing detailed results head: {e}")

    output_csv_path = "./data/results/evaluation_metrics_detailed.csv" # From script
    # Example of saving (ensure directory exists):
    # output_dir = os.path.dirname(output_csv_path)
    # if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    # df_results.to_csv(output_csv_path, index=False)
    # logging.info(f"
💾 Detailed results optionally saved to: {output_csv_path}")

    logging.info("
💡======= Important Notes & Next Steps =======💡")
    # ... (logging messages from script) ...
```

And finally, call the visualization:
```python
# Visualize the results
visualize_results(all_item_metrics, K_VALUES)
```

## 5. Conclusion

This cookbook has walked through a comprehensive RAG evaluation pipeline, from data ingestion to metric calculation and visualization. By implementing the functions and processes described, you can:

1.  **Prepare your data**: Load and process a corpus and a golden dataset.
2.  **Build RAG pipelines**: Construct ingestion and retrieval/generation pipelines using components from the `datapizza-ai` library.
3.  **Evaluate Retrieval Performance**: Calculate exact match, lexical overlap (BLEU/ROUGE on chunks), and similarity-based metrics (Precision@k, Recall@k, F1@k, Hybrid Log-Rank Score) to assess the quality of retrieved context. These directly relate to the metrics discussed in the [Retrieval Evaluation Metrics section of the overview](./evaluation_overview.md#retrieval-evaluation-metrics).
4.  **Evaluate Generation Quality**: Use LLM-as-judge to assess the relevance and accuracy of generated answers, as detailed in the [Generation Evaluation Metrics section](./evaluation_overview.md#generation-evaluation-metrics).
5.  **Analyze Results**: Use visualizations and summary statistics to understand your RAG system's strengths and weaknesses.

Remember that building a robust "Golden Dataset" as described in the [evaluation overview](./evaluation_overview.md#building-a-golden-dataset) is paramount for meaningful evaluation. The choice of metrics should align with your specific application's requirements. This cookbook provides a practical template that you can adapt and extend for your own RAG evaluation tasks.
