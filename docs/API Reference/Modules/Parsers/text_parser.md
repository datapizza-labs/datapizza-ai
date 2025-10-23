# TextParser


<!-- prettier-ignore -->
::: datapizza.modules.parsers.TextParser
    options:
        show_source: false

## Usage

```python
from datapizza.modules.parsers.text_parser import TextParser, parse_text

# Using the class
parser = TextParser()
document_node = parser.parse("Your text content here", metadata={"source": "example"})

# Using the convenience function
document_node = parse_text("Your text content here")
```

## Parameters

The TextParser class takes no initialization parameters.

The `parse` method accepts:
- `text` (str): The text content to parse
- `metadata` (dict, optional): Additional metadata to attach to the document

## Features

- Splits text into paragraphs based on double newlines
- Breaks paragraphs into sentences using regex patterns
- Creates three-level hierarchy: document → paragraphs → sentences
- Preserves original text content in sentence nodes
- Adds index metadata for paragraphs and sentences

## Node Types Created

- `DOCUMENT`: Root document container
- `PARAGRAPH`: Text paragraphs
- `SENTENCE`: Individual sentences with content

## Examples

### Basic Usage

```python
from datapizza.modules.parsers.text_parser import parse_text

text_content = """
This is the first paragraph.
It contains multiple sentences.

This is the second paragraph.
It also has content.
"""

document = parse_text(text_content, metadata={"source": "user_input"})

# Navigate structure
for i, paragraph in enumerate(document.children):
    print(f"Paragraph {i}:")
    for j, sentence in enumerate(paragraph.children):
        print(f"  Sentence {j}: {sentence.content}")
```

### Class-Based Usage

```python
from datapizza.modules.parsers.text_parser import TextParser

parser = TextParser()

# Parse with custom metadata
document = parser.parse(
    text="Sample text content here.",
    metadata={
        "source": "api_input",
        "timestamp": "2024-01-01",
        "language": "en"
    }
)

# Access document metadata
print(f"Source: {document.metadata['source']}")
print(f"Number of paragraphs: {len(document.children)}")
```

### Pipeline Integration

```python
from datapizza.modules.parsers.text_parser import TextParser
from datapizza.modules.splitters import RecursiveSplitter

# Create processing pipeline
parser = TextParser()
splitter = RecursiveSplitter(chunk_size=500)

def process_text_document(text):
    # Parse into hierarchical structure
    document = parser.parse(text)

    # Convert back to flat text for splitting
    full_text = document.content

    # Split into chunks
    chunks = splitter(full_text)

    return document, chunks

# Process document
structured_doc, chunks = process_text_document(long_text)
```

## Best Practices

1. **Use for Simple Text**: Best suited for plain text content without complex formatting
2. **Preprocessing**: Clean text of unwanted characters before parsing if needed
3. **Metadata**: Add relevant metadata during parsing for downstream processing
4. **Pipeline Integration**: Combine with other modules for complete text processing workflows
