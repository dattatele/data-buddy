from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseConnector(ABC):
    """
    Abstract base class for all connectors.
    Provides a common interface for data ingestion.
    """

    @abstractmethod
    def fetch(self, *args: Any, **kwargs: Any) -> Any:
        """
        Fetch data from the source.

        Args:
            *args: Positional arguments specific to the connector.
            **kwargs: Keyword arguments specific to the connector.

        Returns:
            Fetched data in an appropriate format (e.g., pandas DataFrame).
        """
        pass