# Real example: Chatbot

Learn how to build conversational AI applications using the OpenAI client with memory management, context awareness, and advanced chatbot patterns.

## Basic Chatbot


Clients need memory to maintain context and have meaningful conversations. The Memory class stores and manages conversation history, allowing the AI to reference previous exchanges and maintain coherent dialogue.

Here's a simple example of a chatbot with memory:

```python
from datapizza.clients.openai import OpenAIClient
from datapizza.memory import Memory
from datapizza.type import ROLE, TextBlock

client = OpenAIClient(
    api_key="your-api-key",
    model="gpt-4o-mini",
    system_prompt="You are a helpful assistant"
)

def simple_chatbot():
    """Basic chatbot with conversation memory."""

    memory = Memory()

    print("Chatbot: Hello! I'm here to help. Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye!")
            break

        # Get AI response with memory context
        response = client.invoke(user_input, memory=memory)
        print(f"Chatbot: {response.text}")

        # Update conversation memory
        memory.add_turn(TextBlock(content=user_input), role=ROLE.USER)
        memory.add_turn(response.content, role=ROLE.ASSISTANT)

# Run the chatbot
simple_chatbot()
```
