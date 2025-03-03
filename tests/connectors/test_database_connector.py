import pytest
import pandas as pd
from sqlalchemy import create_engine, Engine, Connection
from unittest.mock import MagicMock, patch

from data_warp.connectors.database_connector import DatabaseConnector

def test_init_with_engine():
    """Test that providing an engine sets it on the connector."""
    dummy_engine = MagicMock(spec=Engine)
    connector = DatabaseConnector(engine=dummy_engine)
    assert connector.engine == dummy_engine


@patch("data_warp.connectors.database_connector.create_engine")
def test_init_with_connection_string(mock_create_engine):
    """Test that providing a connection string calls create_engine."""
    dummy_engine = MagicMock(spec=Engine)
    mock_create_engine.return_value = dummy_engine
    connection_string = "sqlite:///:memory:"
    
    connector = DatabaseConnector(connection_string=connection_string)
    
    mock_create_engine.assert_called_once_with(connection_string)
    assert connector.engine == dummy_engine


def test_init_with_no_params():
    """Test that initialization without engine or connection string raises ValueError."""
    with pytest.raises(ValueError, match="Either connection_string or engine must be provided."):
        DatabaseConnector()


@patch("data_warp.connectors.database_connector.pd.read_sql")
def test_fetch_success(mock_read_sql):
    """Test that fetch returns a DataFrame when no errors occur."""
    # Create a dummy connection and context manager for engine.connect()
    dummy_connection = MagicMock(spec=Connection)
    dummy_context_manager = MagicMock()
    dummy_context_manager.__enter__.return_value = dummy_connection
    dummy_context_manager.__exit__.return_value = None

    # Create a dummy engine that returns our context manager on connect()
    dummy_engine = MagicMock(spec=Engine)
    dummy_engine.connect.return_value = dummy_context_manager

    # Set pd.read_sql to return a known DataFrame
    df_expected = pd.DataFrame({"col": [1, 2, 3]})
    mock_read_sql.return_value = df_expected

    connector = DatabaseConnector(engine=dummy_engine)
    query = "SELECT * FROM table"
    df_result = connector.fetch(query)

    # Ensure pd.read_sql was called with the dummy connection and query
    mock_read_sql.assert_called_once_with(query, dummy_connection)
    assert df_result.equals(df_expected)


@patch("data_warp.connectors.database_connector.pd.read_sql")
def test_fetch_failure(mock_read_sql):
    """Test that fetch raises RuntimeError when an exception occurs."""
    dummy_connection = MagicMock(spec=Connection)
    dummy_context_manager = MagicMock()
    dummy_context_manager.__enter__.return_value = dummy_connection
    dummy_context_manager.__exit__.return_value = None

    dummy_engine = MagicMock(spec=Engine)
    dummy_engine.connect.return_value = dummy_context_manager

    # Simulate an exception in pd.read_sql
    mock_read_sql.side_effect = Exception("query error")

    connector = DatabaseConnector(engine=dummy_engine)
    query = "SELECT * FROM table"

    with pytest.raises(RuntimeError, match="Failed to fetch data: query error"):
        connector.fetch(query)
