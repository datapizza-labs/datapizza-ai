import warnings

import pytest
from datapizza.type import Node, NodeType

from datapizza.modules.parsers.docling.docling_parser import DoclingParser
from datapizza.modules.parsers.docling.ocr_options import OCREngine, OCROptions


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
    """Test parse() with deprecated 'pdf_path' and issues warning."""
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
    with pytest.raises(
        ValueError, match="Missing required argument: file_path"
    ):
        parser.parse()


def test_parser_ocr_options_backward_compatibility(mock_docling_parser):
    """Test parser works without explicit OCR options (backward compat)."""
    # Parser created without ocr_options should use default (EasyOCR)
    assert mock_docling_parser.ocr_options.engine == OCREngine.EASY_OCR
    assert (
        mock_docling_parser.ocr_options.easy_ocr_force_full_page is True
    )


def test_parser_with_custom_ocr_options(mock_docling_parser, monkeypatch):
    """Test parser with custom OCR options."""
    custom_options = OCROptions(
        engine=OCREngine.TESSERACT,
        tesseract_lang=["ita"],
    )
    parser = DoclingParser(ocr_options=custom_options)

    assert parser.ocr_options.engine == OCREngine.TESSERACT
    assert parser.ocr_options.tesseract_lang == ["ita"]


def test_parser_with_multilingual_tesseract(mock_docling_parser):
    """Test parser with multiple languages for Tesseract."""
    custom_options = OCROptions(
        engine=OCREngine.TESSERACT,
        tesseract_lang=["ita", "eng", "fra"],
    )
    parser = DoclingParser(ocr_options=custom_options)

    assert parser.ocr_options.engine == OCREngine.TESSERACT
    assert parser.ocr_options.tesseract_lang == ["ita", "eng", "fra"]


def test_parser_with_autodetect_tesseract(mock_docling_parser):
    """Test parser with autodetect for Tesseract."""
    custom_options = OCROptions(
        engine=OCREngine.TESSERACT,
        tesseract_lang=["auto"],
    )
    parser = DoclingParser(ocr_options=custom_options)

    assert parser.ocr_options.engine == OCREngine.TESSERACT
    assert parser.ocr_options.tesseract_lang == ["auto"]


def test_parser_with_ocr_disabled(mock_docling_parser):
    """Test parser with OCR disabled."""
    custom_options = OCROptions(engine=OCREngine.NONE)
    parser = DoclingParser(ocr_options=custom_options)

    assert parser.ocr_options.engine == OCREngine.NONE


def test_parser_preserves_json_output_dir_with_ocr_options(tmp_path):
    """Test parser preserves json_output_dir when using custom OCR options."""
    custom_options = OCROptions(engine=OCREngine.TESSERACT)
    parser = DoclingParser(
        json_output_dir=str(tmp_path),
        ocr_options=custom_options,
    )

    assert parser.json_output_dir == str(tmp_path)
    assert parser.ocr_options.engine == OCREngine.TESSERACT


def test_parse_with_metadata(mock_docling_parser, tmp_path):
    """Test that parse() correctly merges user-provided metadata."""
    dummy_file = tmp_path / "test.pdf"
    dummy_file.write_text("fake-pdf-content")

    user_metadata = {
        "source": "user_upload",
        "custom_field": "test_value",
    }

    node = mock_docling_parser.parse(
        file_path=str(dummy_file), metadata=user_metadata
    )

    assert node.metadata["source"] == "user_upload"
    assert node.metadata["custom_field"] == "test_value"
    # Ensure original metadata is preserved
    assert node.metadata["name"] == "mock_doc"
    assert node.metadata["schema_name"] == "docling_test"


def test_parse_with_none_metadata(mock_docling_parser, tmp_path):
    """Test that parse() works correctly when metadata is None."""
    dummy_file = tmp_path / "test.pdf"
    dummy_file.write_text("fake-pdf-content")

    node = mock_docling_parser.parse(
        file_path=str(dummy_file), metadata=None
    )

    assert isinstance(node, Node)
    assert node.node_type == NodeType.DOCUMENT
    assert node.metadata is not None


def test_parse_metadata_type_validation(mock_docling_parser, tmp_path):
    """Test that parse() raises TypeError for invalid metadata type."""
    dummy_file = tmp_path / "test.pdf"
    dummy_file.write_text("fake-pdf-content")

    with pytest.raises(TypeError, match="metadata must be a dict or None"):
        mock_docling_parser.parse(
            file_path=str(dummy_file), metadata="invalid_string"
        )

    with pytest.raises(TypeError, match="metadata must be a dict or None"):
        mock_docling_parser.parse(file_path=str(dummy_file), metadata=123)

    with pytest.raises(TypeError, match="metadata must be a dict or None"):
        mock_docling_parser.parse(
            file_path=str(dummy_file), metadata=["list", "of", "items"]
        )


def test_parse_metadata_override(mock_docling_parser, tmp_path):
    """Test that user metadata overrides parser-generated metadata."""
    dummy_file = tmp_path / "test.pdf"
    dummy_file.write_text("fake-pdf-content")

    # First, get the default metadata
    node1 = mock_docling_parser.parse(file_path=str(dummy_file))
    original_name = node1.metadata.get("name")
    assert original_name == "mock_doc"

    # Now override with user metadata
    user_metadata = {"name": "custom_name"}
    node2 = mock_docling_parser.parse(
        file_path=str(dummy_file), metadata=user_metadata
    )

    # User metadata should override
    assert node2.metadata["name"] == "custom_name"
    assert node2.metadata["name"] != original_name


def test_parse_with_metadata_and_deprecated_pdf_path(
    mock_docling_parser, tmp_path
):
    """Test metadata works with deprecated pdf_path parameter."""
    dummy_file = tmp_path / "legacy.pdf"
    dummy_file.write_text("fake-pdf")

    user_metadata = {"source": "legacy_path"}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        node = mock_docling_parser.parse(
            pdf_path=str(dummy_file), metadata=user_metadata
        )

        assert isinstance(node, Node)
        assert node.metadata["source"] == "legacy_path"
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
