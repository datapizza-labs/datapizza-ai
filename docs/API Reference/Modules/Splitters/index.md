# Splitters

Splitters are pipeline components that divide large text content into smaller, manageable chunks. They help optimize content for processing, storage, and retrieval in AI applications by creating appropriately sized segments while preserving context and meaning.

## Installation

All splitters are included with `datapizza-ai-core` and require no additional installation.

## Available Splitters

### Core Splitters (Included by Default)

- [RecursiveSplitter](recursive_splitter.md) - Recursively divides text using multiple splitting strategies
- [TextSplitter](text_splitter.md) - Basic text splitter for general-purpose chunking
- [NodeSplitter](node_splitter.md) - Splitter for Node objects preserving hierarchical structure
- [PDFImageSplitter](pdf_image_splitter.md) - Specialized splitter for PDF content with images

## Common Features

- Multiple splitting strategies for different content types
- Configurable chunk sizes and overlap
- Context preservation through overlapping
- Support for structured content (nodes, PDFs, etc.)
- Metadata preservation during splitting
- Spatial layout awareness for document content

## Usage Patterns

### Basic Text Splitting
```python
from datapizza.modules.splitters import RecursiveSplitter

splitter = RecursiveSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter(long_text_content)
```

### Document Processing Pipeline
```python
from datapizza.modules.parsers import TextParser
from datapizza.modules.splitters import NodeSplitter

parser = TextParser()
splitter = NodeSplitter(max_char = 4000)

document = parser.parse(text_content)
structured_chunks = splitter(document)
```

### Choosing the Right Splitter

- **RecursiveSplitter**: Best for general text content, articles, and most use cases
- **TextSplitter**: Simple splitting for basic text without complex requirements
- **NodeSplitter**: When working with structured Node objects from parsers
- **PDFImageSplitter**: Specifically for PDF content with images and complex layouts
- **BBoxMerger**: Utility for processing documents with spatial layout information
