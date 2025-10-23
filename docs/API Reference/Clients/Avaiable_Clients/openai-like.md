


# Openai like


```sh
pip install datapizza-ai-clients-openai-like
```
<!-- prettier-ignore -->
::: datapizza.clients.openai_like.OpenAILikeClient
    options:
        show_source: false

## Key Differences from OpenAIClient

The main difference between OpenAILikeClient and OpenAIClient is the API endpoint they use:

- OpenAILikeClient uses the chat completions API
- OpenAIClient uses the responses API

This makes OpenAILikeClient compatible with services that implement the OpenAI-compatible completions API, such as local models served through Ollama or other providers that follow the OpenAI API specification but only support the completions endpoint.


## Usage example

```python
from datapizza.clients.openai_like import OpenAILikeClient

client = OpenAILikeClient(
    api_key="OPENAI_API_KEY",
    system_prompt="You are a helpful assistant.",
)

response = client.invoke("What is the capital of France?")
print(response.content)
```
