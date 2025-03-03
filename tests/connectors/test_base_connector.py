import pytest
from data_warp.connectors.base_connector import BaseConnector

def test_base_connector_is_abstract():
    """Test that BaseConnector cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseConnector()

def test_base_connector_requires_fetch_implementation():
    """Test that classes inheriting from BaseConnector must implement fetch."""
    class IncompleteConnector(BaseConnector):
        pass

    with pytest.raises(TypeError):
        IncompleteConnector()

def test_base_connector_with_implementation():
    """Test that a proper implementation of BaseConnector works."""
    class ValidConnector(BaseConnector):
        def fetch(self, *args, **kwargs):
            return "data"

    connector = ValidConnector()
    assert connector.fetch() == "data" 