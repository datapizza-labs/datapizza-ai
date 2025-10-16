# Datapizza AI Clients - Regolo

Regolo.ai client for the datapizza-ai framework.

## Installation

```bash
pip install datapizza-ai-clients-regolo
```

## Usage

```python
from datapizza.clients.regolo import RegoloClient
from datapizza.agents import Agent

# Initialize the client
client = RegoloClient(api_key="your-api-key", model="mistral-small3.2")

# Create an agent
agent = Agent(
    name="assistant",
    client=client,
)

# Run the agent
response = agent.run("Hello, how are you?")
print(response)
```

## Configuration

- `api_key`: Your Regolo.ai API key (required)
- `model`: The model to use (default: "mistral-small3.2")

## License

MIT

