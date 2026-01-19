from pathlib import Path

from datapizza.core.models import PipelineComponent
from datapizza.pipeline import IngestionPipeline
from datapizza.type import Chunk, DenseEmbedding, Node

# Get the directory containing this test file for relative path resolution
TEST_DIR = Path(__file__).parent


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


class SplitterWrapper(PipelineComponent):
    """A wrapper component that accepts a splitter element for testing elements feature."""

    def __init__(self, splitter: PipelineComponent):
        self.splitter = splitter

    def _run(self, data):
        """Delegates to the wrapped splitter."""
        return self.splitter(data)

    async def _a_run(self, data):
        """Delegates to the wrapped splitter asynchronously."""
        return await self.splitter.a_run(data)


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

    pipeline = IngestionPipeline().from_yaml(str(TEST_DIR / "config.yaml"))
    assert isinstance(pipeline.pipeline.components[1], LLMCaptioner)
    assert isinstance(pipeline.pipeline.components[2], NodeSplitter)
    assert pipeline.collection_name == "test"


def test_pipeline_from_yaml_with_constants():
    pipeline = IngestionPipeline().from_yaml(str(TEST_DIR / "config.yaml"))
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
        collection_name="test",
    )

    chunks = pipeline.run("Ciao, questo è del testo da ingestionare")

    assert len(chunks) > 0


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

    pipeline = IngestionPipeline(
        modules=[
            TextSplitter(max_char=300),
            FakeEmbedder(),
        ],
        collection_name="test_multi",
    )

    # Test with list of strings
    chunks = pipeline.run(
        ["Primo documento", "Secondo documento", "Terzo documento"],
        metadata={"batch": "test"},
    )

    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"


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

    pipeline = IngestionPipeline(
        modules=[
            TextSplitter(max_char=300),
            FakeEmbedder(),
        ],
        collection_name="test_empty",
    )

    # Should handle empty list without errors
    chunks = pipeline.run([])

    assert len(chunks) == 0, "Should have 0 chunks for empty list"


def test_pipeline_from_yaml_with_elements():
    """Test that elements section in YAML config correctly instantiates and injects components."""
    pipeline = IngestionPipeline().from_yaml(
        str(TEST_DIR / "config_with_elements.yaml")
    )

    # Verify the pipeline has the expected components
    assert len(pipeline.pipeline.components) == 3

    # The second component should be SplitterWrapper with a TextSplitter element
    splitter_wrapper = pipeline.pipeline.components[1]
    # Use class name check because importlib loads from a different module context
    assert splitter_wrapper.__class__.__name__ == "SplitterWrapper"
    assert splitter_wrapper.splitter.__class__.__name__ == "TextSplitter"
    # Verify the element was instantiated with the correct params
    assert splitter_wrapper.splitter.max_char == 2000

    # Verify collection name
    assert pipeline.collection_name == "test_elements"


def test_pipeline_from_yaml_elements_with_constants():
    """Test that elements and constants can coexist and both work correctly."""
    pipeline = IngestionPipeline().from_yaml(
        str(TEST_DIR / "config_with_elements.yaml")
    )

    # Verify constants still work (from the third component - LLMCaptioner)
    captioner = pipeline.pipeline.components[2]
    assert (
        captioner.system_prompt_table
        == "You are a helpful assistant that captions tables."
    )
    assert (
        captioner.system_prompt_figure
        == "You are a helpful assistant that captions figures."
    )


def test_pipeline_from_yaml_backward_compatibility():
    """Test that existing YAML configs without elements section still work."""
    # This uses the original config.yaml which doesn't have elements section
    pipeline = IngestionPipeline().from_yaml(str(TEST_DIR / "config.yaml"))

    # Should work exactly as before
    assert len(pipeline.pipeline.components) == 4
    assert pipeline.collection_name == "test"
