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


def test_parser_ocr_options_backward_compatibility(mock_docling_parser):
    """Test that parser works without explicit OCR options (backward compatible)."""
    # Parser created without ocr_options should use default (EasyOCR)
    assert mock_docling_parser.ocr_options.engine == OCREngine.EASY_OCR
    assert mock_docling_parser.ocr_options.easy_ocr_force_full_page is True


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


