

# Azure Openai

As mentioned in [Microsoft Docs](https://learn.microsoft.com/it-it/azure/ai-foundry/openai/how-to/switching-endpoints?view=foundry-classic), Azure is compatible with the Openai Client


```sh
pip install datapizza-ai-clients-openai
```

## Usage example

```python

from datapizza.clients.openai import OpenAIClient

load_dotenv()
client = OpenAIClient(
    base_url = "https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/"
    api_key= "AZURE_OPENAI_API_KEY",
    model="gpt-4o-mini",
)
response = client.invoke("Hello!")
print(response.text)
```
