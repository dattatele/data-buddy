import pytest
import io
import requests
from unittest.mock import Mock, patch, MagicMock
from data_warp.connectors.sources import (
    LocalFileSource,
    HTTPFileSource,
    S3FileSource,
)

class TestLocalFileSource:
    @patch('os.path.exists')
    def test_fetch_existing_file(self, mock_exists):
        """Test fetching an existing local file."""
        mock_exists.return_value = True
        source = LocalFileSource()
        result = source.fetch("test.csv")
        assert result == "test.csv"

    @patch('os.path.exists')
    def test_fetch_nonexistent_file(self, mock_exists):
        """Test fetching a non-existent local file."""
        mock_exists.return_value = False
        source = LocalFileSource()
        with pytest.raises(FileNotFoundError):
            source.fetch("nonexistent.csv")

    @patch('builtins.open')
    def test_stream_file(self, mock_open):
        """Test streaming a local file."""
        mock_file = Mock()
        mock_file.read.side_effect = [b"chunk1", b"chunk2", b""]
        mock_open.return_value.__enter__.return_value = mock_file
        
        source = LocalFileSource()
        chunks = list(source.stream("test.csv"))
        assert len(chunks) == 2
        assert chunks == [b"chunk1", b"chunk2"]

class TestHTTPFileSource:
    def test_fetch_successful(self):
        """Test successful HTTP fetch."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.content = b"test data"
            mock_get.return_value = mock_response
            
            source = HTTPFileSource()
            result = source.fetch("http://example.com/test.csv")
            assert isinstance(result, io.BytesIO)
            assert result.getvalue() == b"test data"

    def test_fetch_failed(self):
        """Test failed HTTP fetch."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Connection error")
            
            source = HTTPFileSource()
            with pytest.raises(requests.RequestException):
                source.fetch("http://example.com/test.csv")

    def test_stream_successful(self):
        """Test successful HTTP streaming."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value.__enter__.return_value = mock_response
            
            source = HTTPFileSource()
            chunks = list(source.stream("http://example.com/test.csv"))
            assert len(chunks) == 2
            assert chunks == [b"chunk1", b"chunk2"]

    def test_stream_failed(self):
        """Test failed HTTP streaming."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Stream error")
            
            source = HTTPFileSource()
            with pytest.raises(requests.RequestException):
                list(source.stream("http://example.com/test.csv"))

class TestS3FileSource:
    @pytest.fixture
    def mock_s3_client(self):
        with patch('boto3.client') as mock_client:
            yield mock_client.return_value

    def test_fetch_successful(self, mock_s3_client):
        """Test successful S3 fetch."""
        mock_body = Mock()
        mock_body.read.return_value = b"test data"
        mock_s3_client.get_object.return_value = {"Body": mock_body}
        
        source = S3FileSource()
        result = source.fetch("bucket/key.csv")
        assert isinstance(result, io.BytesIO)
        assert result.getvalue() == b"test data"

    def test_fetch_failed(self, mock_s3_client):
        """Test failed S3 fetch."""
        mock_s3_client.get_object.side_effect = Exception("S3 error")
        
        source = S3FileSource()
        with pytest.raises(Exception):
            source.fetch("bucket/key.csv")

    def test_stream_successful(self, mock_s3_client):
        """Test successful S3 streaming."""
        mock_body = Mock()
        mock_body.iter_chunks.return_value = [b"chunk1", b"chunk2"]
        mock_s3_client.get_object.return_value = {"Body": mock_body}
        
        source = S3FileSource()
        chunks = list(source.stream("bucket/key.csv"))
        assert len(chunks) == 2
        assert chunks == [b"chunk1", b"chunk2"]

    def test_fetch_with_invalid_path(self, mock_s3_client):
        """Test S3 fetch with invalid path format."""
        source = S3FileSource()
        with pytest.raises(ValueError):
            source.fetch("invalid-path")

    def test_stream_with_error(self, mock_s3_client):
        """Test S3 streaming with error."""
        mock_s3_client.get_object.side_effect = Exception("Stream error")
        
        source = S3FileSource()
        with pytest.raises(Exception):
            list(source.stream("bucket/key.csv"))

    def test_init_with_credentials(self):
        """Test S3FileSource initialization with credentials."""
        with patch('boto3.client') as mock_client:
            source = S3FileSource(
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret"
            )
            mock_client.assert_called_once_with(
                "s3",
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret"
            ) 