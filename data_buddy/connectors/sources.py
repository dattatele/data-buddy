import os
import io
import logging
from typing import Any, Dict, Iterator, List, Optional, Union
from abc import ABC, abstractmethod

from retrying import retry  
import requests


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileSource(ABC):
    """Abstract base class for different file sources."""

    @abstractmethod
    def fetch(self, file_path: str, **kwargs: Any) -> Union[io.BytesIO, str]:
        """Fetch data from the specified source."""
        pass

    @abstractmethod
    def stream(self, file_path: str, chunk_size: int = 1024 * 1024, **kwargs: Any) -> Any:
        """Stream data from the specified source in chunks."""
        pass


class LocalFileSource(FileSource):
    """Fetch data from local file storage."""

    def fetch(self, file_path: str, **kwargs: Any) -> Union[io.BytesIO, str]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path

    def stream(self, file_path: str, chunk_size: int = 1024 * 1024, **kwargs: Any) -> Any:
        with open(file_path, "rb") as file:
            while chunk := file.read(chunk_size):
                yield chunk


class HTTPFileSource(FileSource):
    """Fetch data from HTTP URLs."""

    @retry(stop_max_attempt_number=3, wait_fixed=2000)  # Retry 3 times with 2-second delay
    def fetch(self, file_path: str, **kwargs: Any) -> Union[io.BytesIO, str]:
        try:
            response = requests.get(file_path, stream=True)
            response.raise_for_status()
            return io.BytesIO(response.content)
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data from {file_path}: {e}")
            raise

    def stream(self, file_path: str, chunk_size: int = 1024 * 1024, **kwargs: Any) -> Any:
        try:
            with requests.get(file_path, stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=chunk_size):
                    yield chunk
        except requests.RequestException as e:
            logger.error(f"Failed to stream data from {file_path}: {e}")
            raise


class S3FileSource(FileSource):
    """Fetch data from AWS S3."""

    def __init__(self, aws_access_key_id: Optional[str] = None, aws_secret_access_key: Optional[str] = None):
        import boto3
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def fetch(self, file_path: str, **kwargs: Any) -> Union[io.BytesIO, str]:
        try:
            bucket, key = file_path.split("/", 1)
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return io.BytesIO(response["Body"].read())
        except Exception as e:
            logger.error(f"Failed to fetch data from S3: {file_path}: {e}")
            raise

    def stream(self, file_path: str, chunk_size: int = 1024 * 1024, **kwargs: Any) -> Any:
        try:
            bucket, key = file_path.split("/", 1)
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            for chunk in response["Body"].iter_chunks(chunk_size):
                yield chunk
        except Exception as e:
            logger.error(f"Failed to stream data from S3: {file_path}: {e}")
            raise

