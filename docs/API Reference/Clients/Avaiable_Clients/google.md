


# Google


```sh
pip install datapizza-ai-clients-google
```
<!-- prettier-ignore -->
::: datapizza.clients.google.GoogleClient
    options:
        show_source: false




# Usage example
```python
import os

from datapizza.clients.google import GoogleClient
from dotenv import load_dotenv

load_dotenv()

client = GoogleClient(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.invoke("Hello!")
print(response.text)
```
