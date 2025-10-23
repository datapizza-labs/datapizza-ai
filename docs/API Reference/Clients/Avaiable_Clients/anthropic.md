# Anthropic


```bash
pip install datapizza-ai-clients-anthropic
```
<!-- prettier-ignore -->
::: datapizza.clients.anthropic.AnthropicClient
    options:
        show_source: false


## Usage example


```python

from datapizza.clients.anthropic import AnthropicClient

client = AnthropicClient(
    api_key="YOUR_API_KEY"
    model="claude-3-5-sonnet-20240620",
)
resposne = client.invoke("hi")
print(response.text)

```

## Show thinking

```python
import os

from datapizza.clients.anthropic import AnthropicClient
from dotenv import load_dotenv

load_dotenv()

client = AnthropicClient(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-sonnet-4-0",
)

response = client.invoke("Hi", thinking =  {"type": "enabled", "budget_tokens": 1024})
print(response)
```
