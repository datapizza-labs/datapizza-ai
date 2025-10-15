# SQLDatabase

```bash
pip install datapizza-ai-tools-sqldatabase
```

<!-- prettier-ignore -->
::: datapizza.tools.SQLDatabase.SQLDatabase
    options:
        show_source: false

## Overview

The SQLDatabase tool provides a powerful interface for AI agents to interact with any SQL database supported by SQLAlchemy. This allows models to query structured, relational data to answer questions, providing more accurate and fact-based responses.

## Features

- **Broad Database Support**: Connect to any database with a SQLAlchemy driver (SQLite, PostgreSQL, MySQL, etc.).
- **Schema Inspection**: Allows the agent to view table schemas to understand data structure before querying.
- **Table Listing**: Lets the agent list all available tables to get context of the database.

## Integration with Agents

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.SQLDatabase import SQLDatabase

db_uri = "sqlite:///company.db"

# 1. Initialize the SQLDatabase tool
db_tool = SQLDatabase(db_uri=db_uri)

# 2. Create an agent and provide it with the database tool's methods
agent = Agent(
    name="database_expert",
    client=OpenAIClient(api_key="YOUR_API_KEY"),
    system_prompt="You are a database expert. Use the available tools to answer questions about the database.",
    tools=[
        db_tool.list_tables,
        db_tool.get_table_schema,
        db_tool.run_sql_query
    ]
)

# 3. Run the agent
response = agent.run("How many people work in the Engineering department?")
print(response)
```
