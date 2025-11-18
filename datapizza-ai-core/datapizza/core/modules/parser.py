from abc import abstractmethod

from datapizza.core.models import PipelineComponent
from datapizza.type import Node


class Parser(PipelineComponent):
    """
    A parser is a pipeline component that converts a document into a structured hierarchical Node representation.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def parse(self, text: str, metadata: dict | None = None) -> Node:
        pass

    async def a_parse(self, text: str, metadata: dict | None = None) -> Node:
        raise NotImplementedError

    def _run(self, text: str, metadata: dict | None = None) -> Node:
        return self.parse(text, metadata)

    async def _a_run(self, text: str, metadata: dict | None = None) -> Node:
        return await self.a_parse(text, metadata)
