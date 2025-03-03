import pandas as pd
from sqlalchemy import create_engine, Engine, Connection


class DatabaseConnector:
    """
    Connector for database-based data sources: MySQL, PostgreSQL, SQLite.
    """

    def __init__(self, connection_string: str = None, engine: Engine = None) -> None:
        """
        Initialize the database connector.

        Args:
            connection_string (str): SQLAlchemy-style connection string.
            engine (Engine): Pre-created SQLAlchemy engine instance.
        """
        if engine:
            self.engine = engine
        elif connection_string:
            self.engine = create_engine(connection_string)
        else:
            raise ValueError("Either connection_string or engine must be provided.")

    def fetch(self, query: str, **kwargs) -> pd.DataFrame:
        """
        Fetch data from the database using an SQL query.

        Args:
            query (str): SQL query string.
            **kwargs: Additional arguments passed to pandas.read_sql.

        Returns:
            pd.DataFrame: DataFrame containing the fetched data.
        """
        try:
            with self.engine.connect() as connection:
                data = pd.read_sql(query, connection, **kwargs)
                print(f"Fetched data using query: {query}")
                return data

        except Exception as e:
            raise RuntimeError(f"Failed to fetch data: {e}")
