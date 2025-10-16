from datapizza.agents.agent import Agent
from datapizza.clients.regolo import RegoloClient
from datapizza.tools import tool
from datapizza.tools.duckduckgo import DuckDuckGoSearchTool

import os

api_key = os.getenv("REGOLO_API_KEY")
if not api_key:
    raise ValueError(
        "REGOLO_API_KEY environment variable not set. "
        "Please set it with: export REGOLO_API_KEY='your-api-key'"
    )

client = RegoloClient(api_key=api_key, model="Llama-3.3-70B-Instruct")

@tool
def get_weather(city: str) -> str:
    return f""" it's sunny all the week in {city}"""

weather_agent = Agent(
    name="weather_expert",
    client=client,
    system_prompt="You are a weather expert. Provide detailed weather information and forecasts.",
    tools=[get_weather],
    max_steps=2,
)

web_search_agent = Agent(
    name="web_search_expert",
    client=client,
    system_prompt="You are a web search expert. You can search the web for information.",
    tools=[DuckDuckGoSearchTool()],
    max_steps=2,
)

planner_agent = Agent(
    name="planner",
    client=client,
    system_prompt="You are a trip planner. You should provide a plan for the user. Make sure to provide a detailed plan with the best places to visit and the best time to visit them.",
    max_steps=2,
)

planner_agent.can_call([weather_agent, web_search_agent])

response = planner_agent.run(
    "I need to plan a hiking trip in Seattle next week. I want to see some waterfalls and a forest."
)
print(response.text)
