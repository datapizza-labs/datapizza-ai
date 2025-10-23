# PDFImageSplitter

<!-- prettier-ignore -->
::: datapizza.modules.splitters.PDFImageSplitter
    options:
        show_source: false


## Usage

```python
from datapizza.modules.splitters import PDFImageSplitter

splitter = PDFImageSplitter()

pdf_chunks = splitter("pdf_path")
```

## Features

- Specialized handling of PDF document structure
- Preserves image data and visual elements
- Maintains spatial layout information
- Includes page-level metadata and coordinates
- Handles complex document layouts with mixed content
- Optimized for PDF content from document intelligence services

## Examples

### Basic PDF Content Splitting

```python
from datapizza.modules.splitters import PDFImageSplitter

# Split while preserving images and layout
pdf_splitter = PDFImageSplitter()

pdf_chunks = pdf_splitter("pdf_path")

# Examine chunks with visual content
for i, chunk in enumerate(pdf_chunks):
    print(f"Chunk {i+1}:")
    print(f"  Content length: {len(chunk.content)}")
    print(f"  Page: {chunk.metadata.get('page_number', 'unknown')}")

    if hasattr(chunk, 'media') and chunk.media:
        print(f"  Media elements: {len(chunk.media)}")
        for media in chunk.media:
            print(f"    Type: {media.media_type}")

    if 'boundingRegions' in chunk.metadata:
        print(f"  Bounding regions: {len(chunk.metadata['boundingRegions'])}")

    print("---")
```
