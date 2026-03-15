


# Google


```sh
pip install datapizza-ai-clients-google
```

<!-- prettier-ignore -->
::: datapizza.clients.google.GoogleClient
    options:
        show_source: false


## Usage example

```python
from datapizza.clients.google import GoogleClient

client = GoogleClient(
    api_key="YOUR_API_KEY",
    model="gemini-2.5-flash",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
)

response = client.invoke("What is the photoelectric effect?")
print(response.text)
```

## Thinking Configuration

Gemini models support explicit reasoning via `thinking_config`. Use `include_thoughts=True` to receive `ThoughtBlock` in the response.

### Gemini 2.5 – Budget-based thinking

```python
response = client.invoke(
    input="Explain step by step why the sky is blue.",
    thinking_config={
        "thinking_budget": 1024,
        "include_thoughts": True,
    },
)

print("Thoughts:", response.thoughts)
print("Answer:", response.text)
```

### Gemini 3 – Level-based thinking

```python
response = client_3.invoke(
    input="Design a simple movie recommendation algorithm.",
    thinking_config={
        "thinking_level": "high",  # "low" | "high"
        "include_thoughts": True,
    },
)

print("Thoughts:", response.thoughts)
print("Answer:", response.text)
```

## Native Tools

The `GoogleClient` supports Gemini's native tools: **Google Search**, **URL Context**, **Google Maps** and **Code Execution**.

### Google Search

```python
from google.genai import types

response = client.invoke(
    input="What are the latest news about the James Webb telescope?",
    tools=[{"google_search": types.GoogleSearch()}],
)
print(response.text)
```

### URL Context

```python
response = client.invoke(
    input="Summarize this page: https://example.com/article",
    tools=[{"url_context": {}}],
)
print(response.text)
```

### Google Maps

```python
response = client.invoke(
    input="List the best restaurants in Rome",
    tools=[{"google_maps": types.GoogleMaps()}],
)
print(response.text)
```

### Code Execution

```python
from google.genai import types

response = client.invoke(
    input="Calculate the sum of the first 50 prime numbers.",
    tools=[{"code_execution": types.ToolCodeExecution()}],
)
print(response.text)
```

## Media Resolution

Control the quality/cost/latency trade-off for multimedia inputs (images, PDFs) with `media_resolution` (Gemini 3 models only).

```python
from google.genai import types

response = client.invoke(
    input="Analyze this technical diagram in detail.",
    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
)
```

Available values: `MEDIA_RESOLUTION_LOW`, `MEDIA_RESOLUTION_MEDIUM`, `MEDIA_RESOLUTION_HIGH`.

## Image Generation

```python
from google.genai import types
from datapizza.type import MediaBlock
import base64

client_img = GoogleClient(
    api_key="YOUR_API_KEY",
    model="gemini-2.5-flash-image",
)

response = client_img.invoke(
    input="Create an image of a cat baking a pizza",
    image_config=types.ImageConfig(aspect_ratio="4:3", image_size="1K"), # Gemini 3 only
)
print(response.text)

# response.content could contain both TextBlock and MediaBlock
for block in response.content:
    if isinstance(block, MediaBlock):
        image_bytes = base64.b64decode(block.media.source)
        with open(f"generated.{block.media.extension}", "wb") as f:
            f.write(image_bytes)
```
