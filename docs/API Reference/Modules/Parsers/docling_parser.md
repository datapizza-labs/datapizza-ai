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
