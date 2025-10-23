
# Mistral


```
pip install datapizza-ai-clients-mistral
```
<!-- prettier-ignore -->
::: datapizza.clients.mistral.MistralClient
    options:
        show_source: false


# Usage Example

```python
from datapizza.clients.mistral import MistralClient

client = MistralClient(
    api_key=os.getenv("MISTRAL_API_KEY"),
    model="mistral-small-latest",
    system_prompt="You are a helpful assistant that responds short and concise.",
)
response = client.invoke("hi")
print(response.text)
```
