from pathlib import Path

from datapizza.clients.mock_client import MockClient
from datapizza.core.clients.client import ClientResponse, StreamInferenceClientModule
from datapizza.core.models import PipelineComponent
from datapizza.embedders.embedders import ClientEmbedder
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline.dag_pipeline import DagPipeline

# Get the directory containing this test file for relative path resolution
TEST_DIR = Path(__file__).parent


class A(PipelineComponent):
    def _run(self, input_data):
        return "A"

    async def _a_run(self):
        return "A"


class B(PipelineComponent):
    def _run(self, input_data):
        return "B"

    async def _a_run(self):
        return "B"


def test_graph_pipeline():
    a = A()
    b = B()

    pipeline = DagPipeline()
    pipeline.add_module("A", a)
    pipeline.add_module("B", b)
    pipeline.connect("A", "B", target_key="input_data")

    assert pipeline.edges[0].from_node_name == "A"
    assert pipeline.edges[0].to_node_name == "B"
    assert pipeline.edges[0].src_key is None
    assert pipeline.edges[0].dst_key == "input_data"

    assert pipeline.nodes == {"A": a, "B": b}


def test_graph_pipeline_run_without_keys():
    a = A()
    b = B()

    pipeline = DagPipeline()
    pipeline.add_module("A", a)
    pipeline.add_module("B", b)
    pipeline.connect("A", "B", target_key="input_data")

    result = pipeline.run({"A": {"input_data": "A"}})

    assert result == {"A": "A", "B": "B"}


def test_graph_pipeline_run_with_target_keys():
    a = A()
    b = B()

    pipeline = DagPipeline()
    pipeline.add_module("A", a)
    pipeline.add_module("B", b)
    pipeline.connect("A", "B", target_key="input_data")

    result = pipeline.run({"A": {"input_data": "A"}})

    assert result == {"A": "A", "B": "B"}


def test_graph_pipeline_run_with_multiple_keys():
    class C(PipelineComponent):
        def _run(self, input_data):
            return {"output_data": "C"}

        async def _a_run(self):
            return {"output_data": "C"}

    a = C()
    b = B()

    pipeline = DagPipeline()
    pipeline.add_module("A", a)
    pipeline.add_module("B", b)
    pipeline.connect("A", "B", source_key="output_data", target_key="input_data")

    result = pipeline.run({"A": {"input_data": "A"}})

    assert result == {"A": {"output_data": "C"}, "B": "B"}


def test_graph_pipeline_from_yaml():
    pipeline = DagPipeline()
    pipeline.from_yaml(str(TEST_DIR / "dag_config.yaml"))

    assert pipeline.nodes["rewriter"].__class__ == ToolRewriter
    assert pipeline.nodes["embedder"].__class__ == ClientEmbedder

    assert pipeline.edges[0].from_node_name == "rewriter"
    assert pipeline.edges[0].to_node_name == "embedder"
    assert pipeline.edges[0].dst_key == "input_data"


def test_graph_pipeline_from_yaml_with_constants():
    pipeline = DagPipeline()
    pipeline.from_yaml(str(TEST_DIR / "dag_config.yaml"))

    assert (
        pipeline.nodes["rewriter"].system_prompt
        == "You are an assistant that answers questions.."
    )


def test_graph_pipeline_with_stream():
    pipeline = DagPipeline()

    client = MockClient(model_name="mock_client")
    generator = StreamInferenceClientModule(client=client)
    pipeline.add_module("generator", generator)

    result = pipeline.run({"generator": {"input": "Hello"}})

    assert isinstance(next(result.get("generator")), ClientResponse)


# def test_graph_pipeline_with_a_stream():
#     pipeline = DagPipeline()
#
#     client = MockClient(model_name="mock_client")
#     generator = StreamInferenceClientModule(client=client)
#     pipeline.add_module("generator", generator)
#     import asyncio
#
#     result = asyncio.run(pipeline.a_run({"generator": {"input": "Hello"}}))
#
#     assert isinstance(result.get("generator"), async_generator)
