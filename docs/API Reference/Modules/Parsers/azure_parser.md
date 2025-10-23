# AzureParser

A document parser that uses Azure AI Document Intelligence to extract structured content from PDFs and other documents.

## Installation

```bash
pip install datapizza-ai-parsers-azure
```

<!-- prettier-ignore -->
::: datapizza.modules.parsers.azure.AzureParser
    options:
        show_source: false




## Usage

```python
from datapizza.modules.parsers.azure import AzureParser

parser = AzureParser(
    api_key="your-azure-key",
    endpoint="https://your-endpoint.cognitiveservices.azure.com/",
    result_type="text"
)

document_node = parser.parse("document.pdf")
```

## Parameters

- `api_key` (str): Azure AI Document Intelligence API key
- `endpoint` (str): Azure service endpoint URL
- `result_type` (str): Output format - "text" or "markdown" (default: "text")

## Features

- Creates hierarchical document structure: document → sections → paragraphs/tables/figures
- Extracts bounding regions and spatial layout information
- Handles tables, figures, and complex document layouts
- Preserves metadata including page numbers and coordinates
- Supports both sync and async processing
- Converts media elements to base64 images with coordinates

## Node Types Created

- `DOCUMENT`: Root document container
- `SECTION`: Document sections
- `PARAGRAPH`: Text paragraphs with content
- `TABLE`: Tables with markdown representation
- `FIGURE`: Images and figures with media data

## Examples

### Basic Document Processing

```python
from datapizza.modules.parsers.azure import AzureParser
import os

parser = AzureParser(
    api_key=os.getenv("AZURE_DOC_INTELLIGENCE_KEY"),
    endpoint=os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT"),
    result_type="markdown"
)

# Parse document
document = parser.parse("complex_document.pdf")

# Access hierarchical structure
for section in document.children:
    for paragraph in section.children:
        print(f"Content: {paragraph.content}")
        print(f"Bounding regions: {paragraph.metadata.get('boundingRegions', [])}")
```

### Async Processing

```python
async def process_document():
    document = await parser.a_run("document.pdf")
    return document

# Usage in async context
document = await process_document()
```
