from pathlib import Path

from datapizza.core.models import PipelineComponent
from datapizza.pipeline import Dependency, FunctionalPipeline

# Get the directory containing this test file for relative path resolution
TEST_DIR = Path(__file__).parent


def test_functional_pipeline():
    class A(PipelineComponent):
        def _run(self):
            return "A"

        async def _a_run(self):
            return "A"

    class B(PipelineComponent):
        def _run(self):
            return "B"

        async def _a_run(self):
            return "B"

    pipeline = FunctionalPipeline()
    pipeline.run("start", A())
    pipeline.run("then", B())
    assert pipeline.execute() == {"start": "A", "then": "B"}


def test_functional_pipeline_with_dependencies():
    class A(PipelineComponent):
        def _run(self):
            return "A"

        async def _a_run(self):
            return "A"

    class B(PipelineComponent):
        def _run(self, data):
            return data + "B"

        async def _a_run(self, data):
            return data + "B"

    pipeline = FunctionalPipeline()
    pipeline.run("start", A())
    pipeline.run(
        "then", B(), dependencies=[Dependency(node_name="start", target_key="data")]
    )
    assert pipeline.execute() == {"start": "A", "then": "AB"}


def test_pipeline_with_then():
    class A(PipelineComponent):
        def _run(self):
            return "A"

        async def _a_run(self):
            return "A"

    class B(PipelineComponent):
        def _run(self, data):
            return "B"

        async def _a_run(self, data):
            return "B"

    pipeline = FunctionalPipeline()
    pipeline.run("start", A())
    pipeline.then("then", B(), target_key="data")
    assert pipeline.execute() == {"start": "A", "then": "B"}


def test_pipeline_with_then_and_dependencies():
    class A(PipelineComponent):
        def _run(self):
            return "A"

        async def _a_run(self):
            return "A"

    class C(PipelineComponent):
        def _run(self):
            return "C"

        async def _a_run(self):
            return "C"

    class B(PipelineComponent):
        def _run(self, data, c):
            return data + "B" + c

        async def _a_run(self, data, c):
            return data + "B" + c

    pipeline = FunctionalPipeline()
    pipeline.run("C", C())  # NOW C MUST BE RUN FIRST of B. In the future, we can
    # make this more flexible
    pipeline.run("start", A())
    pipeline.then(
        "then",
        B(),
        target_key="data",
        dependencies=[Dependency(node_name="C", target_key="c")],
    )
    assert pipeline.execute() == {"start": "A", "then": "ABC", "C": "C"}


def test_pipeline_with_get():
    class A(PipelineComponent):
        def _run(self):
            return "A"

        async def _a_run(self):
            return "A"

    pipeline = FunctionalPipeline().run("start", A()).get("start")
    res = pipeline.execute()

    assert res == "A"


def test_pipeline_from_yaml():
    from datapizza.clients.mock_client import MockClient
    from datapizza.modules.rewriters import ToolRewriter

    pipeline = FunctionalPipeline.from_yaml(
        str(TEST_DIR / "functional_pipeline_config.yaml")
    )
    assert pipeline is not None
    assert pipeline.nodes[0].get("name") == "rewriter"
    assert pipeline.nodes[0].get("node").__class__ == ToolRewriter
    assert isinstance(pipeline.nodes[0].get("node").client, MockClient)
    assert pipeline.nodes[0].get("node").client.model_name == "mock-model"


def test_pipeline_with_foreach():
    class A(PipelineComponent):
        def _run(self):
            return [1, 2, 3]

        async def _a_run(self):
            return [1, 2, 3]

    class B(PipelineComponent):
        def _run(self, data):
            return str(data) + "B"

        async def _a_run(self, data):
            return str(data) + "B"

    pipeline = (
        FunctionalPipeline()
        .run("start", A())
        .foreach(
            "foreach",
            dependencies=[Dependency(node_name="start")],
            do=B(),
        )
    )

    assert pipeline.execute() == {
        "start": [1, 2, 3],
        "foreach": ["1B", "2B", "3B"],
    }


def test_pipeline_with_branch():
    class A(PipelineComponent):
        def _run(self):
            return "A"

        async def _a_run(self):
            return "A"

    class B(PipelineComponent):
        def _run(self):
            return "B"

        async def _a_run(self):
            return "B"

    class C(PipelineComponent):
        def _run(self):
            return "C"

        async def _a_run(self):
            return "C"

    # Test branch with TRUE condition
    pipeline = (
        FunctionalPipeline()
        .run("start", A())
        .branch(
            condition=lambda context: context["start"] == "A",
            if_true=FunctionalPipeline().run("B", B()),
            if_false=FunctionalPipeline().run("C", C()),
        )
    )
    assert pipeline.execute() == {"start": "A", "B": "B"}

    # Test branch with FALSE condition
    pipeline = (
        FunctionalPipeline()
        .run("start", A())
        .branch(
            condition=lambda context: context["start"] == "B",
            if_true=FunctionalPipeline().run("B", B()),
            if_false=FunctionalPipeline().run("C", C()),
        )
    )
    assert pipeline.execute() == {"start": "A", "C": "C"}
