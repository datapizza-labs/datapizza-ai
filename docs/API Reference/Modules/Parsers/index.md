# Parsers

Parsers are pipeline components that convert documents into structured hierarchical Node representations. They extract text, layout information, and metadata from various document formats to create tree-like data structures for further processing.

Each parser should return a [Node](../../Type/node.md) object, which is a hierarchical representation of the document content.

If you write a custom parser that returns a different type of object (for example, the plain text of the document content), you must use a [TreeBuilder](../treebuilder.md) to convert it into a Node.

## Available Parsers

### Core Parsers (Included by Default)

- [TextParser](text_parser.md) - Simple text parser for plain text content

### Optional Parsers (Separate Installation Required)

- [AzureParser](azure_parser.md) - Azure AI Document Intelligence parser for PDFs and documents
- [DoclingParser](docling_parser.md) - Docling-based parser for PDFs with layout preservation and media extraction

## Common Usage Patterns

### Basic Text Processing
```python
from datapizza.modules.parsers.text_parser import parse_text

# Process plain text
document = parse_text("Your text content here")
```

### Document Processing Pipeline
```python
from datapizza.modules.parsers import TextParser
from datapizza.modules.splitters import RecursiveSplitter

# Create processing pipeline
parser = TextParser()
splitter = RecursiveSplitter(chunk_size=1000)

# Process document
document = parser.parse(text_content)
chunks = splitter(document.content)
```
