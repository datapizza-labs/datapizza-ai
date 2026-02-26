# Telegram Bot Tool

Utilities that let datapizza agents interact with the Telegram Bot API. The package wraps common Bot API endpoints —sending messages, photos, documents, and editing previously sent text— into ready-to-use datapizza tools.

## Installation

```bash
pip install datapizza-ai-tools-telegram
```

## Usage

```python
from datapizza.tools.telegram_tool import TelegramBotTool

TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"

client = ...

@tool
def get_weather(location: str, when: str) -> str:
    """Retrieves weather information for a specified location and time."""
    return "25 deg C"

telegram_tool = TelegramBotTool(bot_token=TELEGRAM_TOKEN)
bot_info = telegram_tool.get_me()
print("Bot info:", bot_info)

agent = Agent(name="weather_agent", 
                tools=[
                    get_weather, 
                    telegram_tool.send_message
                    ],
             client=client)
chat_id = ...
agent.run(f"Inviami un messsaggio su Telegram con il meteo di Napoli oggi: {chat_id}")

```

## Testing

```bash
python -m pytest datapizza-ai-tools/telegram/tests
```

Install `pytest` in your environment before running the test suite.
