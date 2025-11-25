
# IBM WatsonX

```sh
pip install datapizza-ai-clients-watsonx
```

## Usage example

```python
from datapizza.clients.watsonx import WatsonXClient

client = WatsonXClient(
    api_key=os.getenv("IBM_API_KEY"),
    url = os.getenv("IBM_ENDPOINT_URL"),
    project_id = os.getenv("IBM_PROJECT_ID"),
    model="meta-llama/llama-3-2-90b-vision-instruct",
)

response = client.invoke("Hello!")
print(response.text)
```
