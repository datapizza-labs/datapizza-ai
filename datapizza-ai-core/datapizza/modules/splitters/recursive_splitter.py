import uuid

from datapizza.core.modules.splitter import Splitter
from datapizza.type.type import Chunk, Node


class RecursiveSplitter(Splitter):
    """
    The RecursiveSplitter takes leaf nodes from a tree document structure and groups them into Chunk objects until reaching the maximum character limit. Each leaf Node represents the smallest unit of content that can be grouped.

    """

    def __init__(self, max_char: int = 5000, overlap: int = 0):
        """
        Initialize the RecursiveSplitter.

        Args:
            max_char: The maximum number of characters per chunk
            overlap: The number of characters to overlap between chunks
        """
        self.max_char = max_char
        self.overlap = overlap

    def _nodes_to_chunk(self, nodes: list[Node]) -> Chunk:
        chunk_id = str(uuid.uuid4())
        chunk_text = " ".join([node.content for node in nodes])
        # check if "boundingRegions" is in the metadata and if it is, merge them
        bounding_regions = [node.metadata.get("boundingRegions", []) for node in nodes]
        chunk_metadata = {
            "boundingRegions": [
                region for regions in bounding_regions for region in regions
            ]
        }
        return Chunk(id=chunk_id, text=chunk_text, metadata=chunk_metadata)

    def get_all_leaves(self, node: Node) -> list[Node]:
        if not node.children:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self.get_all_leaves(child))
        return leaves

    def split(self, node: Node) -> list[Chunk]:
        """
        Split the node into chunks.

        Args:
            node: The node to split

        Returns:
            A list of chunks
        """
        all_leaves = self.get_all_leaves(node)
        result = []
        list_nodes = []
        current_length = 0

        while all_leaves:
            current_node = all_leaves.pop(0)
            node_content_length = len(current_node.content)

            # If adding this node would exceed max_char, create a chunk from collected nodes
            if current_length + node_content_length > self.max_char and list_nodes:
                result.append(self._nodes_to_chunk(list_nodes))
                # Handle overlap if specified
                if self.overlap > 0:
                    # Keep some nodes for overlap in the next chunk
                    overlap_size = 0
                    overlap_nodes = []
                    for i in range(len(list_nodes) - 1, -1, -1):
                        node_len = len(list_nodes[i].content)
                        if overlap_size + node_len <= self.overlap:
                            overlap_nodes.insert(0, list_nodes[i])
                            overlap_size += node_len
                        else:
                            break
                    list_nodes = overlap_nodes
                    current_length = overlap_size
                else:
                    list_nodes = []
                    current_length = 0

            # If the node itself is too large, create a chunk just for it
            if node_content_length > self.max_char:
                if list_nodes:
                    result.append(self._nodes_to_chunk(list_nodes))
                    list_nodes = []
                    current_length = 0
                result.append(self._nodes_to_chunk([current_node]))
            else:
                list_nodes.append(current_node)
                current_length += node_content_length

        # Don't forget remaining nodes
        if list_nodes:
            result.append(self._nodes_to_chunk(list_nodes))

        return result

    async def a_split(self, node: Node) -> list[Chunk]:
        return self.split(node)
