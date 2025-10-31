

# Openai


```sh
pip install datapizza-ai-clients-openai
```
<!-- prettier-ignore -->
::: datapizza.clients.openai.OpenAIClient
    options:
        show_source: false



## Usage example

```python

from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(
    api_key="YOUR_API_KEY",
    model="gpt-4o-mini",
)
response = client.invoke("Hello!")
print(response.text)
```


## Include thinking
```python
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(
    model= "gpt-5",
    api_key="YOUR_API_KEY",
)

response = client.invoke("Hi",reasoning={
        "effort": "low",
        "summary": "auto"
    }
)
print(response)
```
