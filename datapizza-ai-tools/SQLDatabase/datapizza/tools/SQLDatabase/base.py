import json

from sqlalchemy import create_engine, inspect, text

from datapizza.tools import tool


class SQLDatabase:
    """
    A collection of tools to interact with a SQL database using SQLAlchemy.
    This class is a container for methods that are exposed as tools.
    """

    def __init__(self, db_uri: str):
        """
        Initializes the SQLDatabase tool container.

        Args:
            db_uri (str): The database URI for connection (e.g., "sqlite:///my_database.db").
        """
        self.engine = create_engine(db_uri)

    @tool
    def list_tables(self) -> str:
        """
        Returns a newline-separated string of available table names in the database.
        """
        inspector = inspect(self.engine)
        return "\n".join(inspector.get_table_names())

    @tool
    def get_table_schema(self, table_name: str) -> str:
        """
        Returns the schema of a specific table in a human-readable format.

        Args:
            table_name: The name of the table to inspect.
        """
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        schema_description = f"Schema for table '{table_name}':\n"
        for col in columns:
            schema_description += f"  - {col['name']} ({col['type']})\n"
        return schema_description.strip()

    @tool
    def run_sql_query(self, query: str) -> str:
        """
        Executes a SQL query and returns the result.
        For SELECT statements, it returns a JSON string of the rows.
        For other statements (INSERT, UPDATE, DELETE), it returns a success message.

        Args:
            query: The SQL query to execute.
        """
        with self.engine.connect() as connection:
            trans = connection.begin()
            try:
                result = connection.execute(text(query))
                if result.returns_rows:
                    rows = [dict(row._mapping) for row in result.fetchall()]
                    trans.commit()
                    return json.dumps(rows, indent=2)
                else:
                    trans.commit()
                    return f"Query '{query}' executed successfully. {result.rowcount} rows affected."
            except Exception as e:
                trans.rollback()
                return f"Error executing query: {e}"
