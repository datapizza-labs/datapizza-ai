from datapizza.modules.splitters.node_splitter import NodeSplitter
from datapizza.type.type import Node


def test_process_within_limit():
    """Test processing a node within character limit"""
    splitter = NodeSplitter(max_char=100)
    content = "Short content"
    node = Node(content=content)
    result = splitter.run(node)
    assert len(result) == 1
    assert result[0].text == content


def test_process_with_children():
    """Test processing a node with children"""
    splitter = NodeSplitter(max_char=10)
    parent = Node(content="Parent node")
    child1 = Node(content="Child 1")
    child2 = Node(content="Child 2")
    parent.children = [child1, child2]

    result = splitter.run(parent)
    assert len(result) == 2
    assert any(node.text == "Child 1" for node in result)
    assert any(node.text == "Child 2" for node in result)


def test_empy_node():
    splitter = NodeSplitter(max_char=10)
    node = Node()
    result = splitter.run(node)
    assert len(result) == 0


def test_process_large_content_no_children():
    """Test processing a node with content larger than max_char but no children"""
    splitter = NodeSplitter(max_char=10)
    content = "This is a very long content that exceeds the maximum character limit"
    node = Node(content=content)
    result = splitter.run(node)
    assert len(result) == 1  # Should return original node if no way to split
    assert result[0].text == content
