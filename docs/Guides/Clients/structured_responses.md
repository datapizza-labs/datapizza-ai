# Structured Responses

Generate structured, typed data from AI responses using Pydantic models. This ensures consistent output format and enables easy data validation.

## Basic Usage

### Simple Model

```python
from datapizza.clients.openai import OpenAIClient
from pydantic import BaseModel

client = OpenAIClient(api_key="your-api-key", model="gpt-4o-mini")

class Person(BaseModel):
    name: str
    age: int
    occupation: str

response = client.structured_response(
    input="Create a profile for a software engineer",
    output_cls=Person
)

person = response.structured_data[0]
print(f"Name: {person.name}")
print(f"Age: {person.age}")
print(f"Occupation: {person.occupation}")
```

### Complex Models

```python
from typing import List
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str
    price: float = Field(gt=0, description="Price must be positive")
    tags: List[str]
    in_stock: bool

class Store(BaseModel):
    name: str
    location: str
    products: List[Product]

response = client.structured_response(
    input="Create a tech store with 3 products",
    output_cls=Store
)

store = response.structured_data[0]
print(f"Store: {store.name}")
for product in store.products:
    print(f"- {product.name}: ${product.price}")
```

## Data Extraction

### Extract Information from Text

```python
class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str
    company: str

text = """
Hi, I'm John Smith from TechCorp.
You can reach me at john.smith@techcorp.com or call 555-0123.
"""

response = client.structured_response(
    input=f"Extract contact information from this text: {text}",
    output_cls=ContactInfo
)

contact = response.structured_data[0]
print(f"Contact: {contact.name} at {contact.company}")
```

### Analyze and Categorize

```python
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class TextAnalysis(BaseModel):
    sentiment: Sentiment
    confidence: float = Field(ge=0, le=1)
    key_topics: List[str]
    summary: str

review = "This product is amazing! Great quality and fast shipping."

response = client.structured_response(
    input=f"Analyze this review: {review}",
    output_cls=TextAnalysis
)

analysis = response.structured_data[0]
print(f"Sentiment: {analysis.sentiment}")
print(f"Confidence: {analysis.confidence}")
print(f"Topics: {', '.join(analysis.key_topics)}")
```
