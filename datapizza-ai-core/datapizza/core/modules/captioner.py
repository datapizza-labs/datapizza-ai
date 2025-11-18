from abc import abstractmethod

from datapizza.core.models import PipelineComponent
from datapizza.type import Node


class NodeCaptioner(PipelineComponent):
    """
    A captioner that can caption a node.
    """

    @abstractmethod
    def caption(self, node: Node):
        """
        Caption a node.
        """
        pass

    async def a_caption(self, node: Node):
        """
        async Caption a node.
        """
        raise NotImplementedError

    def _run(self, node: Node):
        return self.caption(node)

    async def _a_run(self, node: Node):
        return await self.a_caption(node)


class Captioner(PipelineComponent):
    """
    A captioner that can caption a node.
    """

    @abstractmethod
    def caption(self, node: Node):
        """
        Caption a node.

        Args:
            node: The node to caption.

        Returns:
            The same node with the caption.
        """
        pass

    async def a_caption(self, node: Node):
        raise NotImplementedError

    def _run(self, node: Node):
        return self.caption(node)

    async def _a_run(self, node: Node):
        return await self.a_caption(node)
