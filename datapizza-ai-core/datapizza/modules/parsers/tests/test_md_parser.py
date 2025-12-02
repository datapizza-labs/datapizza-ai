import os
import tempfile

import pytest

from datapizza.modules.parsers.md_parser import MDParser
from datapizza.type import NodeType


@pytest.fixture
def temp_md_file():
    content = """# Title

This is a paragraph.

## Section 1

This is section 1.

### Subsection 1.1

Content 1.1.

## Section 2

Content 2.
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        f.write(content)
        path = f.name
    yield path
    os.remove(path)


def test_md_parser_hierarchy(temp_md_file):
    parser = MDParser()
    root = parser.parse(temp_md_file, metadata={"source": "test"})

    assert root.node_type == NodeType.DOCUMENT
    assert root.metadata["file_path"] == temp_md_file
    assert root.metadata["source"] == "test"
    assert len(root.children) == 1  # # Title

    title_node = root.children[0]
    assert title_node.node_type == NodeType.SECTION
    assert title_node.metadata["title"] == "Title"
    assert title_node.metadata["level"] == 1
    assert title_node.metadata["file_path"] == temp_md_file

    # Title content
    assert len(title_node.children) == 3  # Paragraph, Section 1, Section 2
    # Wait, logic check:
    # H1 Title
    #   Para
    #   H2 Section 1
    #     Para
    #     H3 Subsection 1.1
    #       Para
    #   H2 Section 2
    #     Para

    # Let's check the structure created by the parser.
    # stack starts with [(0, doc)]
    # # Title (level 1). pop until stack[-1][0] < 1. 0 < 1. ok. stack: [(0, doc), (1, Title)]
    # Para "This is a paragraph." -> child of Title.
    # ## Section 1 (level 2). pop until stack[-1][0] < 2. 1 < 2. ok. stack: [(0, doc), (1, Title), (2, Section 1)]
    # Para "This is section 1." -> child of Section 1.
    # ### Subsection 1.1 (level 3). pop until stack[-1][0] < 3. 2 < 3. ok. stack: [(0, doc), (1, Title), (2, Sec 1), (3, Sub 1.1)]
    # Para "Content 1.1." -> child of Sub 1.1.
    # ## Section 2 (level 2). pop until stack[-1][0] < 2. (3, Sub 1.1) pop. (2, Sec 1) pop. (1, Title) < 2. ok. stack: [(0, doc), (1, Title), (2, Section 2)]
    # Para "Content 2." -> child of Section 2.

    # So Title should have 3 children?
    # Children of Title:
    # 1. Paragraph
    # 2. Section 1
    # 3. Section 2 (since it was popped back to Title)

    assert len(title_node.children) == 3

    para1 = title_node.children[0]
    assert para1.node_type == NodeType.PARAGRAPH
    assert para1.children[0].content == "This is a paragraph."
    assert para1.metadata["file_path"] == temp_md_file

    sec1 = title_node.children[1]
    assert sec1.node_type == NodeType.SECTION
    assert sec1.metadata["title"] == "Section 1"
    assert sec1.metadata["file_path"] == temp_md_file

    # Section 1 children: Paragraph and Subsection 1.1
    assert len(sec1.children) == 2
    assert sec1.children[0].node_type == NodeType.PARAGRAPH
    assert sec1.children[0].children[0].content == "This is section 1."

    subsec1 = sec1.children[1]
    assert subsec1.node_type == NodeType.SECTION
    assert subsec1.metadata["title"] == "Subsection 1.1"

    sec2 = title_node.children[2]
    assert sec2.node_type == NodeType.SECTION
    assert sec2.metadata["title"] == "Section 2"


def test_md_parser_no_headers():
    parser = MDParser()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        f.write("Just a paragraph.")
        path = f.name

    try:
        root = parser.parse(path)
        assert root.node_type == NodeType.DOCUMENT
        assert root.metadata["file_path"] == path
        assert len(root.children) == 1
        assert root.children[0].node_type == NodeType.PARAGRAPH
        assert root.children[0].children[0].content == "Just a paragraph."
        assert root.children[0].metadata["file_path"] == path
    finally:
        os.remove(path)


def test_md_parser_sentences():
    parser = MDParser()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        f.write("Sentence one. Sentence two.")
        path = f.name

    try:
        root = parser.parse(path)
        para = root.children[0]
        assert len(para.children) == 2
        assert para.children[0].content == "Sentence one."
        assert para.children[1].content == "Sentence two."
        assert para.children[0].metadata["file_path"] == path
    finally:
        os.remove(path)
