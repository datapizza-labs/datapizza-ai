import json

import pytest

from datapizza.tools.SQLDatabase.base import SQLDatabase


@pytest.fixture
def db_tool() -> SQLDatabase:
    """Provides a SQLDatabase instance connected to an in-memory SQLite database."""
    db = SQLDatabase(db_uri="sqlite:///:memory:")
    setup_queries = [
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(50), age INTEGER);",
        "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30);",
        "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25);",
    ]
    for query in setup_queries:
        db.run_sql_query(query)
    return db


def test_list_tables(db_tool: SQLDatabase):
    """Tests that list_tables returns the correct table names."""
    tables = db_tool.list_tables()
    assert isinstance(tables, str)
    assert "users" in tables.split("\n")


def test_get_table_schema(db_tool: SQLDatabase):
    """Tests that get_table_schema returns the correct schema information."""
    schema = db_tool.get_table_schema("users")
    assert isinstance(schema, str)
    assert "Schema for table 'users':" in schema
    assert "id (INTEGER)" in schema
    assert "name (VARCHAR(50))" in schema
    assert "age (INTEGER)" in schema


def test_run_select_query(db_tool: SQLDatabase):
    """Tests that run executes a SELECT query and returns correct data."""
    result = db_tool.run_sql_query("SELECT name, age FROM users WHERE id = 1;")

    data = json.loads(result)

    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0] == {"name": "Alice", "age": 30}


def test_run_insert_query(db_tool: SQLDatabase):
    """Tests that run executes an INSERT statement and data is added."""
    insert_query = "INSERT INTO users (id, name, age) VALUES (3, 'Charlie', 35);"
    result = db_tool.run_sql_query(insert_query)
    assert "1 rows affected" in result

    select_result = db_tool.run_sql_query("SELECT * FROM users WHERE id = 3;")
    data = json.loads(select_result)
    assert len(data) == 1
    assert data[0]["name"] == "Charlie"


def test_run_query_error(db_tool: SQLDatabase):
    """Tests that a malformed query returns an error message."""
    result = db_tool.run_sql_query("SELECT * FROM non_existent_table;")
    assert isinstance(result, str)
    assert "Error executing query:" in result
