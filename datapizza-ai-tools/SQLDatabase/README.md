<div align="center">
<img src="https://github.com/datapizza-labs/datapizza-ai/raw/main/docs/assets/logo_bg_dark.png" alt="Datapizza AI Logo" width="200" height="200">

# Datapizza AI - SQLDatabase Tool

**A tool for Datapizza AI that allows agents to interact with SQL databases using SQLAlchemy.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

This tool provides a robust and easy-to-use interface for connecting `datapizza-ai` agents to any SQL database supported by SQLAlchemy (including SQLite, PostgreSQL, MySQL, and more).

Agents equipped with this tool can inspect database schemas, list tables, and execute SQL queries to answer questions based on structured data.

> **⚠️ Warning: Risk of Data Modification**
> 
> Using queries like `INSERT`, `UPDATE`, and `DELETE` will permanently modify the data in your database. Exercise extreme caution. Before performing write operations in a production environment, consider the following:
> - Test queries in a development or staging environment.
> - Use a "query-writing" agent to generate and validate the SQL before execution.
> - Ensure you have recent backups of your database.

## ⚙️ How it Works

The `SQLDatabase` tool is a class that, once initialized with a database URI, exposes three distinct functionalities to an agent:

1.  `list_tables()`: Lists all tables available in the database. Returns a newline-separated string of table names.
2.  `get_table_schema(table_name: str)`: Retrieves the schema for a specified table. Returns a formatted string describing the columns and their data types.
3.  `run_sql_query(query: str)`: Executes a given SQL query. For `SELECT` statements, it returns a JSON-formatted string of the results. For other operations (e.g., `INSERT`, `UPDATE`), it returns a success message.

## 🚀 Quick Start

### 1. Installation

```bash
# Install the core framework
pip install datapizza-ai

# Install the SQLDatabase tool
pip install datapizza-ai-tools-sqldatabase
```

> **Note on Database Drivers:**
> 
> This tool uses SQLAlchemy, which requires specific DB-API drivers to connect to different databases. For example, if you want to connect to PostgreSQL or MySQL, you'll need to install their respective drivers:
> 
> ```bash
> # For PostgreSQL
> pip install psycopg2-binary
> 
> # For MySQL
> pip install mysql-connector-python
> ```
> 
> Please refer to the [SQLAlchemy documentation](https://docs.sqlalchemy.org/en/20/dialects/) for a full list of supported databases and their required drivers.

### 2. Example: Creating a Database Expert Agent

In this example, we'll create an agent that can answer questions about a simple SQLite database.

```python
import sqlite3
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.SQLDatabase import SQLDatabase

# ---
# Setup: Create a dummy database for the example
# In a real scenario, you would connect to your existing database.
# ---
db_uri = "sqlite:///company.db"

# Clean up previous runs if the file exists
try:
    import os
    os.remove("company.db")
except OSError:
    pass

# Create and populate the database
conn = sqlite3.connect('company.db')
cursor = conn.cursor()
cursor.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER)")
cursor.execute("INSERT INTO employees (name, department, salary) VALUES ('Alice', 'Engineering', 80000)")
cursor.execute("INSERT INTO employees (name, department, salary) VALUES ('Bob', 'HR', 65000)")
cursor.execute("INSERT INTO employees (name, department, salary) VALUES ('Charlie', 'Engineering', 95000)")
conn.commit()
conn.close()
# ---
# End of Setup
# ---


# 1. Initialize the SQLDatabase tool with the database URI
db_tools = SQLDatabase(db_uri=db_uri)

# 2. Initialize a client (e.g., OpenAI)
client = OpenAIClient(api_key="YOUR_API_KEY")

# 3. Create an agent and provide it with the database tools
agent = Agent(
    name="database_expert",
    client=client,
    system_prompt="""You are an expert and careful SQL database assistant. Your primary goal is to answer questions about the database by executing queries.

Follow these steps:
1.  Use `list_tables` to identify the relevant tables.
2.  Use `get_table_schema` to understand the columns and data types of those tables before writing a query.
3.  Construct an efficient SQL query to answer the user's question.
4.  Execute the query using `run_sql_query`.
5.  Analyze the results and provide a clear, human-readable answer to the user.

**Important:** Be extra cautious with `UPDATE` and `DELETE` operations. Ensure you understand the request correctly before modifying data.""",
    tools=[
        db_tools.list_tables,
        db_tools.get_table_schema,
        db_tools.run_sql_query
    ]
)

# 4. Run the agent to answer questions
print("--- Query 1: What are the available tables? ---")
response = agent.run("What tables are in the database?")
print(f"Agent Response: {response.text}\n")

print("--- Query 2: How many employees are in the Engineering department? ---")
response = agent.run("How many employees work in the Engineering department?")
print(f"Agent Response: {response.text}\n")

print("--- Query 3: Who is the highest-paid employee? ---")
response = agent.run("Who is the highest-paid employee and what is their salary?")
print(f"Agent Response: {response.text}\n")

print("--- Query 4: Update Bob's salary ---")
response = agent.run("Update Bob's salary to 70000.")
print(f"Agent Response: {response.text}\n")

print("--- Query 5: Add a new employee ---")
response = agent.run("Add a new employee named 'David' in the 'Sales' department with a salary of 75000.")
print(f"Agent Response: {response.text}\n")

print("--- Query 6: Delete an employee ---")
response = agent.run("Remove the employee named 'Bob' from the database.")
print(f"Agent Response: {response.text}\n")

```

### Expected Output:

```
--- Query 1: What are the available tables? ---
Agent Response: The database contains the following table:

- **employees**

--- Query 2: How many employees are in the Engineering department? ---
Agent Response: There are 2 employees working in the Engineering department.

--- Query 3: Who is the highest-paid employee? ---
Agent Response: The highest-paid employee is Charlie, with a salary of $95,000.

--- Query 4: Update Bob's salary ---
Agent Response: Bob's salary has been successfully updated to 70000.

--- Query 5: Add a new employee ---
Agent Response: I have successfully added David to the employees table.

--- Query 6: Delete an employee ---
Agent Response: The employee named Bob has been removed from the database.
```
