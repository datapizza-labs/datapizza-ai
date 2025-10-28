from datapizza.vectorstores.qdrant import QdrantVectorstore

from datapizza.core.models import PipelineComponent
from datapizza.core.vectorstore import VectorConfig
from datapizza.pipeline import IngestionPipeline
from datapizza.type import Chunk, DenseEmbedding, Node


class CustomSplitter(PipelineComponent):
    def __init__(self):
        pass

    def _run(self, nodes: list[Node]):
        """Takes a list of Nodes and returns them (dummy implementation)."""
        if not isinstance(nodes, list):
            raise TypeError(f"Expected input to be a list of Nodes, got {type(nodes)}")
        return nodes

    async def _a_run(self, nodes: list[Node]):
        """Takes a list of Nodes and returns them (dummy implementation)."""
        if not isinstance(nodes, list):
            raise TypeError(f"Expected input to be a list of Nodes, got {type(nodes)}")
        return nodes


def test_pipeline():
    pipeline = IngestionPipeline(
        modules=[
            CustomSplitter(),
        ]
    )
    assert len(pipeline.pipeline.components) == 1
    assert isinstance(pipeline.pipeline.components[0], CustomSplitter)


def test_pipeline_from_yaml():
    from datapizza.modules.captioners.llm_captioner import LLMCaptioner
    from datapizza.modules.splitters.node_splitter import NodeSplitter

    pipeline = IngestionPipeline().from_yaml(
        "datapizza-ai-core/datapizza/pipeline/tests/config.yaml"
    )
    assert isinstance(pipeline.pipeline.components[1], LLMCaptioner)
    assert isinstance(pipeline.pipeline.components[2], NodeSplitter)
    assert isinstance(pipeline.vector_store, QdrantVectorstore)
    assert pipeline.collection_name == "test"


def test_pipeline_from_yaml_with_constants():
    pipeline = IngestionPipeline().from_yaml(
        "datapizza-ai-core/datapizza/pipeline/tests/config.yaml"
    )
    assert (
        pipeline.pipeline.components[1].system_prompt_table
        == "You are a helpful assistant that captions tables."
    )
    assert (
        pipeline.pipeline.components[1].system_prompt_figure
        == "You are a helpful assistant that captions figures."
    )


def test_ingestion_pipeline():
    from datapizza.modules.splitters.text_splitter import TextSplitter

    qdrant_client = QdrantVectorstore(location=":memory:")
    qdrant_client.create_collection(
        "test",
        vector_config=[VectorConfig(name="embedding", dimensions=1536)],
    )

    class FakeEmbedder(PipelineComponent):
        def _run(self, nodes: list[Chunk]):
            for node in nodes:
                node.embeddings = [
                    DenseEmbedding(name="embedding", vector=[0.0] * 1536)
                ]
            return nodes

        async def _a_run(self, nodes: list[Chunk]):
            return self._run(nodes)

    pipeline = IngestionPipeline(
        modules=[
            TextSplitter(max_char=300),
            FakeEmbedder(),
        ],
        vector_store=qdrant_client,
        collection_name="test",
    )

    pipeline.run("Ciao, questo è del testo da ingestionare")

    data = qdrant_client.search(
        collection_name="test",
        query_vector=[0.0] * 1536,
        k=1,
    )

    assert len(data) == 1


def test_ingestion_pipeline_with_multiple_files():
    """Test issue #63: IngestionPipeline should handle list of file paths."""
    from datapizza.modules.splitters.text_splitter import TextSplitter

    class FakeEmbedder(PipelineComponent):
        def _run(self, nodes: list[Chunk]):
            for node in nodes:
                node.embeddings = [
                    DenseEmbedding(name="embedding", vector=[0.0] * 1536)
                ]
            return nodes

    qdrant_client = QdrantVectorstore(location=":memory:")
    qdrant_client.create_collection(
        "test_multi",
        vector_config=[VectorConfig(name="embedding", dimensions=1536)],
    )

    pipeline = IngestionPipeline(
        modules=[
            TextSplitter(max_char=300),
            FakeEmbedder(),
        ],
        vector_store=qdrant_client,
        collection_name="test_multi",
    )

    # Test with list of strings
    pipeline.run(
        ["Primo documento", "Secondo documento", "Terzo documento"],
        metadata={"batch": "test"},
    )

    data = qdrant_client.search(
        collection_name="test_multi",
        query_vector=[0.0] * 1536,
        k=10,
    )

    assert len(data) == 3, f"Expected 3 chunks, got {len(data)}"


def test_ingestion_pipeline_list_validation():
    """Test that invalid list elements are rejected."""
    from datapizza.modules.splitters.text_splitter import TextSplitter

    pipeline = IngestionPipeline(
        modules=[TextSplitter(max_char=300)],
    )

    # Should raise ValueError for non-string elements
    try:
        pipeline.run([123, "valid_string", None])
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "must be strings" in str(e).lower()


def test_ingestion_pipeline_empty_list():
    """Test that empty list is handled gracefully."""
    from datapizza.modules.splitters.text_splitter import TextSplitter

    class FakeEmbedder(PipelineComponent):
        def _run(self, nodes: list[Chunk]):
            for node in nodes:
                node.embeddings = [
                    DenseEmbedding(name="embedding", vector=[0.0] * 1536)
                ]
            return nodes

    qdrant_client = QdrantVectorstore(location=":memory:")
    qdrant_client.create_collection(
        "test_empty",
        vector_config=[VectorConfig(name="embedding", dimensions=1536)],
    )

    pipeline = IngestionPipeline(
        modules=[
            TextSplitter(max_char=300),
            FakeEmbedder(),
        ],
        vector_store=qdrant_client,
        collection_name="test_empty",
    )

    # Should handle empty list without errors
    pipeline.run([])

    data = qdrant_client.search(
        collection_name="test_empty",
        query_vector=[0.0] * 1536,
        k=10,
    )

    assert len(data) == 0, "Should have 0 chunks for empty list"
