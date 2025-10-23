# Client Factory

The ClientFactory provides a convenient way to create LLM clients for different providers without having to import and instantiate each client type individually.


<!-- prettier-ignore -->
::: datapizza.clients.factory.ClientFactory
    options:
        show_source: false



## Example Usage

```python
from datapizza.clients.factory import ClientFactory, Provider

# Create an OpenAI client
openai_client = ClientFactory.create(
    provider=Provider.OPENAI,
    api_key="OPENAI_API_KEY",
    model="gpt-4",
    system_prompt="You are a helpful assistant.",
    temperature=0.7
)

# Create a Google client using string provider
google_client = ClientFactory.create(
    provider="google",
    api_key="GOOGLE_API_KEY",
    model="gemini-pro",
    system_prompt="You are a helpful assistant.",
    temperature=0.5
)

# Create an Anthropic client with custom parameters
anthropic_client = ClientFactory.create(
    provider=Provider.ANTHROPIC,
    api_key="ANTHROPIC_API_KEY",
    model="claude-3-sonnet-20240229",
    system_prompt="You are a helpful assistant.",
    temperature=0.3,
)

# Use the client
response = openai_client.invoke("What is the capital of France?")
print(response.content)
```

## Supported Providers

- `openai` - OpenAI GPT models
- `google` - Google Gemini models
- `anthropic` - Anthropic Claude models
- `mistral` - Mistral AI models
