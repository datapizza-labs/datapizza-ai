# Captioners

Captioners are pipeline components that generate captions and descriptions for media content such as images, figures, and tables. They use LLM clients to analyze visual content and produce descriptive text.

### LLMCaptioner

<!-- prettier-ignore -->
::: datapizza.modules.captioners.LLMCaptioner
    options:
        show_source: false


A captioner that uses language models to generate captions for media nodes (figures and tables) within document hierarchies.

```python
from datapizza.clients.openai import OpenAIClient
from datapizza.modules.captioners import LLMCaptioner
from datapizza.type import ROLE, Media, MediaNode, NodeType

client = OpenAIClient(api_key="OPENAI_API_KEY", model="gpt-4o")
captioner = LLMCaptioner(
    client=client,
    max_workers=3,
    system_prompt_table="Describe this table in detail.",
    system_prompt_figure="Describe this figure/image in detail."
)

document_node = MediaNode( node_type=NodeType.FIGURE, children=[], metadata={}, media=Media(source_type="path", source="gogole.png", extension="png", media_type="image"))
captioned_document = captioner(document_node)
print(captioned_document)
```

**Parameters:**

- `client` (Client): The LLM client to use for caption generation
- `max_workers` (int): Maximum number of concurrent workers for parallel processing (default: 3)
- `system_prompt_table` (str, optional): System prompt for table captioning
- `system_prompt_figure` (str, optional): System prompt for figure captioning

**Features:**

- Automatically finds all media nodes (figures and tables) in a document hierarchy
- Generates captions using configurable system prompts
- Supports concurrent processing for better performance
- Creates new paragraph nodes containing the original content plus generated captions
- Preserves original node metadata and structure
- Supports both sync and async processing

**Supported Node Types:**

- `FIGURE`: Images and visual figures
- `TABLE`: Tables and tabular data

**Output Format:**

The captioner creates new paragraph nodes with content in the format:
```
{original_content} <{node_type}> [{generated_caption}]
```
