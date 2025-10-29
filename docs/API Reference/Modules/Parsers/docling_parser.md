# DoclingParser

A document parser that uses Docling to convert PDF files into structured hierarchical Node representations with preserved layout information and media extraction.

## Installation

```bash
pip install datapizza-ai-parsers-docling
```

<!--
::: datapizza.modules.parsers.docling.DoclingParser
    options:
        show_source: false


 -->



## Usage

```python
from datapizza.modules.parsers.docling import DoclingParser

# Basic usage
parser = DoclingParser()
document_node = parser.parse("sample.pdf")

print(document_node)
```

## Parameters

- `json_output_dir` (str, optional): Directory to save intermediate Docling JSON results for debugging and inspection

## Features

- **PDF Processing**: Converts PDF files using Docling's DocumentConverter with OCR and table structure detection
- **Hierarchical Structure**: Creates logical document hierarchy (document → sections → paragraphs/tables/figures)
- **Media Extraction**: Extracts images and tables as base64-encoded media with bounding box coordinates
- **Layout Preservation**: Maintains spatial layout information including page numbers and bounding regions
- **Markdown Generation**: Converts tables to markdown format and handles list structures
- **Metadata Rich**: Preserves full Docling metadata in `docling_raw` with convenience fields

## Configuration

The parser automatically configures Docling with:

- Table structure detection enabled
- Full page OCR with EasyOCR
- PyPdfium backend for PDF processing

## Examples

### Basic Document Processing

```python
from datapizza.modules.parsers.docling import DoclingParser

parser = DoclingParser()
document = parser.parse("research_paper.pdf")

# Access hierarchical structure
for section in document.children:
    print(f"Section: {section.metadata.get('docling_label', 'Unknown')}")
    for child in section.children:
        if child.node_type.name == "PARAGRAPH":
            print(f"  Paragraph: {child.content[:100]}...")
        elif child.node_type.name == "TABLE":
            print(f"  Table with {len(child.children)} rows")
        elif child.node_type.name == "FIGURE":
            print(f"  Figure: {child.metadata.get('docling_label', 'Image')}")
```

### Configured OCR Document Processing

```python
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.parsers.docling.ocr_options import OCROptions, OCREngine

# Configure parser with EasyOCR (default, backward compatible)
parser = DoclingParser()
document = parser.parse("document.pdf")

# Configure parser with Tesseract OCR for Italian language
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["ita"],
)
parser = DoclingParser(ocr_options=ocr_config)
document = parser.parse("italian_document.pdf")

# Configure parser with Tesseract for multiple languages
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["eng", "fra"],  # English and French
)
parser = DoclingParser(ocr_options=ocr_config)
document = parser.parse("multilingual_document.pdf")

# Configure parser with Tesseract for Italian and English
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["ita", "eng"],
)
parser = DoclingParser(ocr_options=ocr_config)
document = parser.parse("italian_english_document.pdf")

# Configure parser with automatic language detection
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["auto"],
)
parser = DoclingParser(ocr_options=ocr_config)
document = parser.parse("mixed_language_document.pdf")

# Disable OCR completely (for documents without text that needs OCR)
ocr_config = OCROptions(engine=OCREngine.NONE)
parser = DoclingParser(ocr_options=ocr_config)
document = parser.parse("native_text_document.pdf")

# Enable custom EasyOCR configuration
ocr_config = OCROptions(
    engine=OCREngine.EASY_OCR,
    easy_ocr_force_full_page=False,  # Process only text-light regions
)
parser = DoclingParser(ocr_options=ocr_config)
document = parser.parse("document.pdf")

# Parse with JSON output for debugging
ocr_config = OCROptions(engine=OCREngine.TESSERACT, tesseract_lang=["ita", "eng"])
parser = DoclingParser(
    json_output_dir="./docling_debug",
    ocr_options=ocr_config,
)
document = parser.parse("document.pdf")
# Intermediate Docling JSON will be saved to ./docling_debug/document.json
```

### Tesseract Language Options

When using Tesseract OCR, you can specify languages in the `tesseract_lang` parameter as a list:

**Single Language:**
```python
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["eng"],  # English
)
```

**Multiple Languages:**
```python
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["eng", "ita", "fra"],  # English, Italian, French
)
```

**Automatic Language Detection:**
```python
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["auto"],  # Auto-detect language
)
```

**Common Language Codes:**
- `"eng"` - English
- `"ita"` - Italian
- `"fra"` - French
- `"deu"` - German
- `"spa"` - Spanish
- `"por"` - Portuguese
- `"chi_sim"` - Simplified Chinese
- `"chi_tra"` - Traditional Chinese
- `"jpn"` - Japanese
- `"auto"` - Automatic detection

For a complete list of supported languages, refer to [Tesseract documentation](https://github.com/UB-Mannheim/tesseract/wiki).
