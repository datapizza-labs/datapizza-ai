import warnings

import pytest
from datapizza.type import Node, NodeType

from datapizza.modules.parsers.docling.docling_parser import DoclingParser


def test_parse_with_file_path(mock_docling_parser, tmp_path):
    """Test that parse() works with the new 'file_path' parameter."""
    dummy_file = tmp_path / "dummy.pdf"
    dummy_file.write_text("fake-pdf-content")

    node = mock_docling_parser.parse(file_path=str(dummy_file))
    assert isinstance(node, Node)
    assert node.node_type == NodeType.DOCUMENT
    assert node.metadata["name"] == "mock_doc"
    assert node.metadata["schema_name"] == "docling_test"


def test_parse_with_pdf_path_deprecated(mock_docling_parser, tmp_path):
    """Test that parse() still works with deprecated 'pdf_path' and issues a warning."""
    dummy_file = tmp_path / "legacy.pdf"
    dummy_file.write_text("fake-pdf")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        node = mock_docling_parser.parse(pdf_path=str(dummy_file))

        assert isinstance(node, Node)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "pdf_path" in str(w[0].message)


def test_parse_with_both_file_and_pdf_path(mock_docling_parser, tmp_path):
    """Ensure file_path takes precedence if both are given."""
    dummy_file1 = tmp_path / "primary.pdf"
    dummy_file1.write_text("pdf1")
    dummy_file2 = tmp_path / "secondary.pdf"
    dummy_file2.write_text("pdf2")

    with warnings.catch_warnings(record=True) as w:
        node = mock_docling_parser.parse(
            file_path=str(dummy_file1), pdf_path=str(dummy_file2)
        )

        # Expect a warning but use file_path
        assert len(w) == 1
        assert node.metadata["name"] == "mock_doc"


def test_parse_missing_file_path_raises():
    parser = DoclingParser()
    with pytest.raises(ValueError, match="Missing required argument: file_path"):
        parser.parse()
