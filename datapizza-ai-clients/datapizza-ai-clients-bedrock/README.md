# DataPizza AI - AWS Bedrock Client

AWS Bedrock client implementation for the datapizza-ai framework. This client provides seamless integration with AWS Bedrock's Converse API, supporting various foundation models including Anthropic's Claude models.

## Features

- Full support for AWS Bedrock Converse API
- Multiple authentication methods (AWS Profile, Access Keys, Environment Variables)
- Streaming and non-streaming responses
- Tool/function calling support
- Memory/conversation history management
- Image and document (PDF) support
- Async support

## Installation

```bash
pip install datapizza-ai-clients-bedrock
```

Or install from source in editable mode:

```bash
cd datapizza-ai/datapizza-ai-clients/datapizza-ai-clients-bedrock
pip install -e .
```

## Quick Start

### Basic Usage

```python
from datapizza.clients.bedrock import BedrockClient

# Using AWS Profile
client = BedrockClient(
    profile_name="my-aws-profile",
    region_name="us-east-1"
)

# Or using access keys
client = BedrockClient(
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    region_name="us-east-1"
)

# Simple invocation
result = client.invoke("What is AWS Bedrock?")

# Extract text from response
for block in result.content:
    if hasattr(block, 'content'):
        print(block.content)
```

## Authentication Methods

The client supports multiple authentication methods in the following priority order:

### 1. Explicit Credentials

```python
client = BedrockClient(
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    aws_session_token="YOUR_SESSION_TOKEN",  # Optional, for temporary credentials
    region_name="us-east-1"
)
```

### 2. AWS Profile

```python
client = BedrockClient(
    profile_name="my-aws-profile",
    region_name="us-east-1"
)
```

### 3. Environment Variables

Set these environment variables:
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_SESSION_TOKEN="your-session-token"  # Optional
export AWS_PROFILE="my-aws-profile"  # Or use profile
```

Then initialize without parameters:
```python
client = BedrockClient(region_name="us-east-1")
```

### 4. Default AWS Credentials Chain

If no credentials are provided, boto3 will use the default credentials chain (IAM roles, ~/.aws/credentials, etc.)

```python
client = BedrockClient(region_name="us-east-1")
```

## Available Models

The client works with any Bedrock model that supports the Converse API. Popular models include:

- `anthropic.claude-3-5-sonnet-20241022-v2:0` (default)
- `anthropic.claude-3-5-sonnet-20240620-v1:0`
- `anthropic.claude-3-opus-20240229-v1:0`
- `anthropic.claude-3-sonnet-20240229-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`
- `meta.llama3-70b-instruct-v1:0`
- `mistral.mistral-large-2402-v1:0`
- And many more...

```python
client = BedrockClient(
    model="anthropic.claude-3-opus-20240229-v1:0",
    region_name="us-east-1"
)
```

## Usage Examples

### With System Prompt

```python
client = BedrockClient(
    system_prompt="You are a helpful coding assistant specialized in Python.",
    region_name="us-east-1"
)

result = client.invoke("How do I read a CSV file?")
```

### Streaming Responses

```python
for chunk in client.stream_invoke("Tell me a long story"):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
print()
```

### With Memory (Conversation History)

```python
from datapizza.memory import Memory

memory = Memory()
client = BedrockClient(region_name="us-east-1")

# First message
result1 = client.invoke("My favorite color is blue", memory=memory)

# The conversation is tracked in memory
result2 = client.invoke("What's my favorite color?", memory=memory)
# Response: "Your favorite color is blue."
```

### With Temperature and Max Tokens

```python
result = client.invoke(
    "Write a creative story",
    temperature=0.9,  # Higher = more creative (0-1)
    max_tokens=1000
)
```

### With Tools/Function Calling

```python
from datapizza.tools import Tool

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the weather for a location"""
    return f"The weather in {location} is 22°{unit[0].upper()}"

weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a location",
    function=get_weather,
    properties={
        "location": {
            "type": "string",
            "description": "The city name"
        },
        "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit"
        }
    },
    required=["location"]
)

result = client.invoke(
    "What's the weather in Paris?",
    tools=[weather_tool]
)

# Check for function calls
for block in result.content:
    if isinstance(block, FunctionCallBlock):
        print(f"Function: {block.name}")
        print(f"Arguments: {block.arguments}")
```

### Async Support

```python
import asyncio

async def main():
    client = BedrockClient(region_name="us-east-1")
    result = await client.a_invoke("Hello!")
    print(result.content[0].content)

asyncio.run(main())
```

### Async Streaming

```python
async def stream_example():
    client = BedrockClient(region_name="us-east-1")
    async for chunk in client.a_stream_invoke("Count to 10"):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)

asyncio.run(stream_example())
```

## Configuration

### Constructor Parameters

```python
BedrockClient(
    model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    system_prompt: str = "",
    temperature: float | None = None,  # 0-1 for most models
    cache: Cache | None = None,
    region_name: str = "us-east-1",
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    profile_name: str | None = None,
)
```

### Invoke Parameters

```python
client.invoke(
    input: str,                    # The user message
    tools: list[Tool] | None = None,
    memory: Memory | None = None,
    tool_choice: "auto" | "required" | "none" | list[str] = "auto",
    temperature: float | None = None,
    max_tokens: int = 2048,
    system_prompt: str | None = None,  # Override instance system_prompt
)
```

## Response Format

All methods return a `ClientResponse` object:

```python
response = client.invoke("Hello")

# Access content blocks
for block in response.content:
    if isinstance(block, TextBlock):
        print(block.content)  # The text
    elif isinstance(block, FunctionCallBlock):
        print(block.name)      # Function name
        print(block.arguments) # Function arguments

# Token usage
print(f"Prompt tokens: {response.prompt_tokens_used}")
print(f"Completion tokens: {response.completion_tokens_used}")
print(f"Stop reason: {response.stop_reason}")
```

## IAM Permissions

Your AWS credentials need the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/*"
            ]
        }
    ]
}
```

## Model Access

Before using a model, you need to request access in the AWS Bedrock console:

1. Go to AWS Bedrock console
2. Navigate to "Model access"
3. Request access to the models you want to use
4. Wait for approval (usually instant for most models)

## Limitations

- Structured responses are not natively supported (unlike OpenAI's structured output)
- Some advanced features may vary by model
- Token usage metrics may not include caching information

## Error Handling

```python
from botocore.exceptions import BotoCoreError, ClientError

try:
    result = client.invoke("Hello")
except ClientError as e:
    if e.response['Error']['Code'] == 'AccessDeniedException':
        print("Model access not granted. Check Bedrock console.")
    elif e.response['Error']['Code'] == 'ResourceNotFoundException':
        print("Model not found in this region.")
    else:
        print(f"AWS Error: {e}")
except BotoCoreError as e:
    print(f"Boto3 Error: {e}")
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Formatting

```bash
ruff check .
ruff format .
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please see the main datapizza-ai repository for contribution guidelines.

## Support

For issues and questions:
- GitHub Issues: [datapizza-ai repository](https://github.com/datapizza/datapizza-ai)
- Documentation: [DataPizza AI Docs](https://docs.datapizza.ai)
