"""
OCR (Optical Character Recognition) configuration for Docling parser.

This module provides flexible OCR engine management, allowing users to:
- Select between different OCR engines (EasyOCR, Tesseract)
- Disable OCR entirely
- Configure engine-specific options

Backward compatibility: Default is EasyOCR if no options are provided.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OCREngine(str, Enum):
    """
    Supported OCR engines for document processing.

    Attributes:
        EASY_OCR: Use EasyOCR for text recognition
        TESSERACT: Use Tesseract OCR engine
        NONE: Disable OCR - use only PDF text extraction
    """

    EASY_OCR = "easy_ocr"
    TESSERACT = "tesseract"
    NONE = "none"

    def __str__(self) -> str:
        """Return human-readable name for the OCR engine."""
        return {
            self.EASY_OCR: "EasyOCR",
            self.TESSERACT: "Tesseract",
            self.NONE: "None",
        }[self]


@dataclass
class OCROptions:
    """
    Configuration for OCR processing in Docling.

    This dataclass allows fine-grained control over OCR behavior during
    document parsing. Each OCR engine has its own set of parameters.

    Attributes:
        engine: The OCR engine to use (default: EASY_OCR for backward compatibility)
        easy_ocr_force_full_page: Force full page OCR with EasyOCR (default: True)
        tesseract_lang: Language codes for Tesseract as list (default: ["eng"])
                       Examples: ["auto"], ["ita"], ["ita", "eng"], ["eng", "fra"]
        tesseract_config: Additional Tesseract configuration string (default: "")
    """

    engine: OCREngine = field(default=OCREngine.EASY_OCR)
    # EasyOCR specific options
    easy_ocr_force_full_page: bool = field(default=True)
    # Tesseract specific options
    tesseract_lang: list[str] = field(default_factory=lambda: ["eng"])
    tesseract_config: str = field(default="")

    def to_docling_pipeline_options(self) -> dict[str, Any]:
        """
        Convert OCR options to Docling PdfPipelineOptions configuration.

        This method translates our internal OCR configuration to the format
        expected by Docling's DocumentConverter.

        Returns:
            Dictionary of pipeline kwargs suitable for PdfPipelineOptions.
            Always includes "do_table_structure": True for consistency.
        """
        pipeline_kwargs: dict[str, Any] = {"do_table_structure": True}

        if self.engine == OCREngine.NONE:
            # No OCR - Docling will use built-in PDF text extraction only
            return pipeline_kwargs

        if self.engine == OCREngine.EASY_OCR:
            # Import here to avoid dependency issues if EasyOCR not installed
            from docling.datamodel.pipeline_options import EasyOcrOptions

            ocr_options = EasyOcrOptions(
                force_full_page_ocr=self.easy_ocr_force_full_page
            )
            pipeline_kwargs["ocr_options"] = ocr_options
            return pipeline_kwargs

        if self.engine == OCREngine.TESSERACT:
            # Import here to support optional Tesseract dependency
            from docling.datamodel.pipeline_options import TesseractOcrOptions

            # tesseract_lang is already a list, pass it directly
            ocr_options = TesseractOcrOptions(lang=self.tesseract_lang)
            pipeline_kwargs["ocr_options"] = ocr_options
            return pipeline_kwargs

        # Fallback (should not reach here if Enum is exhaustive)
        return pipeline_kwargs

