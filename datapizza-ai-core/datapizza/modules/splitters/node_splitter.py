from datapizza.core.modules.splitter import Splitter
from datapizza.type.type import Chunk, Node


class NodeSplitter(Splitter):
    """
    A splitter that traverses a document tree from the root node. If the root node's content is smaller than max_chars,
    it becomes a single chunk. Otherwise, it recursively processes the node's children, creating chunks from the first
    level of children that fit within max_chars, continuing deeper into the tree structure as needed.
    """

    def __init__(self, max_char: int = 5000):
        """
        Initialize the NodeSplitter.

        Args:
            max_char: The maximum number of characters per chunk
        """
        self.max_char = max_char

    def _node_to_chunks(self, nodes: list[Node]) -> list[Chunk]:
        return [
            Chunk(id=str(node.id), text=node.content, metadata=node.metadata)
            for node in nodes
            if node.content
        ]

    def split(self, node: Node) -> list[Chunk]:
        """
        Split the node into chunks.

        Args:
            node: The node to split

        Returns:
            A list of chunks
        """
        if len(node.content) <= self.max_char:
            return self._node_to_chunks([node])

        result = []

        for child in node.children:
            result.extend(self.split(node=child))

        if not result:
            return self._node_to_chunks([node])

        return result

    def __call__(self, node: Node) -> list[Chunk]:
        return self.split(node)

    async def a_split(self, node: Node) -> list[Chunk]:
        return self.split(node)
