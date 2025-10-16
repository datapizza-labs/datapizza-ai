from datapizza.clients.regolo import RegoloClient
from datapizza.embedders.regolo import RegoloEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

import os

# Get Regolo API key from environment
regolo_api_key = os.getenv("REGOLO_API_KEY")
if not regolo_api_key:
    raise ValueError(
        "REGOLO_API_KEY environment variable not set. "
        "Please set it with: export REGOLO_API_KEY='your-api-key'"
    )

regolo_client = RegoloClient(
    model="gpt-oss-120b",
    api_key=regolo_api_key
)

dag_pipeline = DagPipeline()
dag_pipeline.add_module(
    "rewriter",
    ToolRewriter(
        client=regolo_client,
        system_prompt=(
            "Rewrite user queries to improve retrieval accuracy."
        ),
        tool_choice="auto"  # Changed from "required" to "auto" for compatibility
    )
)
dag_pipeline.add_module(
    "embedder",
    RegoloEmbedder(
        api_key=regolo_api_key,
        model_name="Qwen3-Embedding-8B"
    )
)
dag_pipeline.add_module(
    "retriever",
    QdrantVectorstore(host="localhost", port=6333).as_retriever(
        collection_name="my_documents",
        k=5
    )
)
dag_pipeline.add_module(
    "prompt",
    ChatPromptTemplate(
        user_prompt_template="User question: {{user_prompt}}\n:",
        retrieval_prompt_template=(
            "Retrieved content:\n"
            "{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}"
        )
    )
)
dag_pipeline.add_module("generator", regolo_client.as_module_component())

dag_pipeline.connect("rewriter", "embedder", target_key="text")
dag_pipeline.connect("embedder", "retriever", target_key="query_vector")
dag_pipeline.connect("retriever", "prompt", target_key="chunks")
dag_pipeline.connect("prompt", "generator", target_key="memory")

query = "tell me something about this document"

# Check if Qdrant is available before running
try:
    from qdrant_client import QdrantClient
    qdrant_test = QdrantClient(host="localhost", port=6333)
    collections = qdrant_test.get_collections()
    print(f"✓ Qdrant connected. Collections: {[c.name for c in collections.collections]}")
    
    result = dag_pipeline.run({
        "rewriter": {"user_prompt": query},
        "prompt": {"user_prompt": query},
        "retriever": {"collection_name": "my_documents", "k": 3},
        "generator":{"input": query}
    })
    
    print(f"Generated response: {result['generator']}")
except Exception as e:
    print(f"⚠️  Error: {e}")
    print("Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
    print("And that 'my_documents' collection exists with embedded documents.")