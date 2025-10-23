# Multimodality

The clients supports various media types including images and PDFs, allowing you to create rich multimodal applications.

## Supported Media Types

| Media Type | Supported Formats | Source Types |
|------------|------------------|--------------|
| Images | PNG, JPEG, GIF, WebP | File path, URL, base64 |
| PDFs | PDF documents | File path, base64 |

## Basic Image Input

### Single Image from File

```python
from datapizza.clients.openai import OpenAIClient
from datapizza.type import Media, MediaBlock, TextBlock

client = OpenAIClient(
    api_key="your-api-key",
    model="gpt-4o"  # Vision models required for images
)

# Create image media object
image = Media(
    media_type="image",
    source_type="path",
    source="image.png", # Use the correct path
    extension="png"
)

# Create media block
media_block = MediaBlock(media=image)
text_block = TextBlock(content="What do you see in this image?")

# Send multimodal input
response = client.invoke(
    input=[text_block, media_block],
    max_tokens=200
)

print(response.text)
```

### Image from URL

```python
# Image from URL
image_url = Media(
    media_type="image",
    source_type="url",
    source="https://example.com/image.png",
    extension="png"
)

response = client.invoke(
    input=[
        TextBlock(content="Describe this image"),
        MediaBlock(media=image_url)
    ]
)
print(response.text)
```

### Image from Base64

```python
import base64

# Read and encode image
with open("image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

image_b64 = Media(
    media_type="image",
    source_type="base64",
    source=base64_image,
    extension="png"
)

response = client.invoke(
    input=[
        TextBlock(content="Analyze this image"),
        MediaBlock(media=image_b64)
    ]
)
print(response.text)
```

## Multiple Images

Compare or analyze multiple images in a single request:

```python
# Multiple images for comparison
image1 = Media(
    media_type="image",
    source_type="path",
    source="before.png",
    extension="png"
)

image2 = Media(
    media_type="image",
    source_type="path",
    source="after.png",
    extension="png"
)

response = client.invoke(
    input=[
        TextBlock(content="Compare these two images and describe the differences"),
        MediaBlock(media=image1),
        MediaBlock(media=image2)
    ],
    max_tokens=300
)

print(response.text)
```

## Working with PDFs

```python
# PDF from file path
pdf_doc = Media(
    media_type="pdf",
    source_type="path",
    source="document.pdf",
    extension="pdf"
)

response = client.invoke(
    input=[
        TextBlock(content="Summarize the key points from this document"),
        MediaBlock(media=pdf_doc)
    ],
    max_tokens=500
)

print(response.text)
```


## Working with Audio


Google handle audio inline

```sh
pip install datapizza-ai-clients-google
```

```python
from datapizza.clients.google import GoogleClient
from datapizza.type import Media, MediaBlock, TextBlock

client = GoogleClient(
    api_key="YOUR_API_KEY",
    model="gemini-2.0-flash-exp"
)
# PDF from file path
media = Media(
    media_type="audio",
    source_type="path",
    source="sample.mp3",
    extension="mp3"
)

response = client.invoke(
    input=[
        TextBlock(content="Summarize the key points from this audio file"),
        MediaBlock(media=media)
    ],
)

print(response.text)
```
