import os
import io
import csv
import json
import time
import logging
from typing import Callable, Any, Dict, List, Union, Optional, Iterator
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import functools
import inspect

import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.json as pajson
import pyarrow.parquet as pq
import pyarrow.parquet as pq

from retrying import retry  # External library for retry mechanism
from data_warp.connectors.base_connector import BaseConnector
from data_warp.connectors.sources import FileSource, LocalFileSource, HTTPFileSource, S3FileSource
from data_warp.connectors.utils import inherit_docstring_and_signature, StreamingBatchIterator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping of readers to their respective functions
READER_FUNCTIONS = {
    "csv": {
        "pandas": pd.read_csv,
        "builtin": csv.DictReader,
        "pyarrow": pacsv.read_csv,
    },
    "json": {
        "pandas": pd.read_json,
        "builtin": json.load,
        "pyarrow": pajson.read_json,
    },
    "parquet": {
        "pandas": pd.read_parquet,
        "pyarrow": pq.read_table
    }
    # Add more file types and readers as needed
}

class FileConnector(BaseConnector):
    """
    Universal connector for file-based data sources: CSV, JSON, Parquet, etc.
    Supports local, HTTP, AWS S3, GCP Cloud Storage, and Azure Blob Storage.
    """

    def __init__(
        self,
        file_path: str,
        file_type: Optional[str] = None,
        reader: str = "pandas",
        source: str = "local",
        chunk_size: Optional[int] = None,
        streaming: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the file connector.

        Args:
            file_path (str): Path to the file or URL.
            file_type (str): Type of file ('csv', 'json', 'parquet').
            reader (str): Reader to use ('pandas' or 'builtin').
            source (str): Source of the file ('local', 'http', 's3', 'gcp', 'azure').
            chunk_size (Optional[int]): Number of rows to read per chunk (for large files in pandas).
            streaming (bool): If True, enables line-by-line/record-by-record streaming methods (for CSV/JSON).
            **kwargs: Additional arguments for the file source.
        """
        self.file_path = file_path
        self.file_type = file_type.lower() if file_type else self._infer_file_type(file_path)
        self.reader = reader.lower()
        self.source = source.lower()
        self.chunk_size = chunk_size
        self.streaming = streaming
        self.source_handler = self._get_source_handler(source, **kwargs)

    def _infer_file_type(self, file_path: str) -> str:
        """Infer file type from the file extension."""
        return Path(file_path).suffix.lstrip('.').lower()

    def _get_source_handler(self, source: str, **kwargs: Any) -> FileSource:
        """Get the appropriate file source handler."""
        handlers = {
            "local": LocalFileSource(),
            "http": HTTPFileSource(),
            "s3": S3FileSource(**kwargs)
        }
        if source not in handlers:
            raise ValueError(f"Unsupported file source: {source}")
        return handlers[source]

    # ------------------ CSV Fetching Methods ------------------ #
    def _fetch_csv(self, file_obj: Union[str, io.BytesIO], **kwargs: Any) -> Union[pd.DataFrame, List[Dict]]:
        """Fetch data from a CSV file."""
        if self.reader == "pandas":
            return self._fetch_csv_pandas(file_obj, **kwargs)
        elif self.reader == "builtin":
            return self._fetch_csv_builtin(file_obj, **kwargs)
        elif self.reader == "pyarrow":
            return self._fetch_csv_pyarrow(file_obj, **kwargs)
        else:
            raise ValueError(f"Unsupported reader for CSV: {self.reader}")

    @inherit_docstring_and_signature(pd.read_csv)
    def _fetch_csv_pandas(self, file_obj: Union[str, io.BytesIO], **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(file_obj, **kwargs)
    
    @inherit_docstring_and_signature(csv.DictReader)
    def _fetch_csv_builtin(self, file_obj: Union[str, io.BytesIO], **kwargs: Any) -> List[Dict]:
        # Assumes file_path is a local file.
        with open(self.file_path, mode='r', newline='') as file:
            return list(csv.DictReader(file, **kwargs))
    
    @inherit_docstring_and_signature(pacsv.read_csv)
    def _fetch_csv_pyarrow(self, file_obj: Union[str, io.BytesIO], **kwargs: Any):
        return pacsv.read_csv(file_obj, **kwargs)
    
    # ------------------ Excel Fetching Methods ------------------ #
        
    def _fetch_excel(self, file_obj: Union[str, io.BytesIO], **kwargs: Any) -> pd.DataFrame:
        if self.reader == "pandas":
            return self._fetch_excel_pandas(file_obj, **kwargs)
        elif self.reader == "builtin":
            raise ValueError("Builtin reader is not supported for Excel files. Use 'pandas' reader instead.")
        else:
            raise ValueError(f"Unsupported reader for Excel: {self.reader}")
    
    @inherit_docstring_and_signature(pd.read_excel)
    def _fetch_excel_pandas(self, file_obj: Union[str, io.BytesIO], **kwargs: Any) -> pd.DataFrame:
        return pd.read_excel(file_obj, **kwargs)
    
    # ------------------ JSON Fetching Methods ------------------ #

    def _fetch_json(self, file_obj: Union[str, io.BytesIO], **kwargs: Any) -> Union[pd.DataFrame, dict, list]:
        if self.reader == "pandas":
            return self._fetch_json_pandas(file_obj, **kwargs)
        elif self.reader == "builtin":
            return self._fetch_json_builtin(file_obj, **kwargs)
        elif self.reader == "pyarrow":
            return self._fetch_json_pyarrow(file_obj, **kwargs)
        else:
            raise ValueError(f"Unsupported reader for JSON: {self.reader}")
    
    @inherit_docstring_and_signature(pd.read_json)
    def _fetch_json_pandas(self, file_obj: Union[str, io.BytesIO], **kwargs: Any) -> pd.DataFrame:
        return pd.read_json(file_obj, **kwargs)
    
    @inherit_docstring_and_signature(json.load)
    def _fetch_json_builtin(self, file_obj: Union[str, io.BytesIO], **kwargs: Any) -> Union[dict, list]:
        if isinstance(file_obj, io.BytesIO):
            return json.load(file_obj)
        with open(file_obj, 'r') as file:
            return json.load(file)
    
    @inherit_docstring_and_signature(pajson.read_json)
    def _fetch_json_pyarrow(self, file_obj: Union[str, io.BytesIO], **kwargs: Any):
        return pajson.read_json(file_obj, **kwargs)
    
    # ------------------ Parquet Fetching Methods ------------------ #

    def _fetch_parquet(self, file_obj: Union[str, io.BytesIO], **kwargs: Any) -> pd.DataFrame:
        if self.reader == "pandas":
            return self._fetch_parquet_pandas(file_obj, **kwargs)
        elif self.reader == "pyarrow":
            return self._fetch_parquet_pyarrow(file_obj, **kwargs)
        else:
            raise ValueError(f"Unsupported reader for Parquet: {self.reader}")
    
    @inherit_docstring_and_signature(pd.read_parquet)
    def _fetch_parquet_pandas(self, file_obj: Union[str, io.BytesIO], **kwargs: Any) -> pd.DataFrame:
        kwargs.setdefault('use_threads', True)
        kwargs.setdefault('memory_map', True)
        return pd.read_parquet(file_obj, **kwargs)
    
    @inherit_docstring_and_signature(pq.read_table)
    def _fetch_parquet_pyarrow(self, file_obj: Union[str, io.BytesIO], **kwargs: Any):
        return pq.read_table(file_obj, **kwargs)
    
    # ------------------ Helper Functions for JSON Streaming ------------------

    def _is_dict_of_lists(self, data: dict) -> bool:
        """
        Return True if data is a dict whose values are all lists of equal length.
        """
        if not data:
            return False
        values = list(data.values())
        if not all(isinstance(v, list) for v in values):
            return False
        lengths = [len(v) for v in values]
        return len(set(lengths)) == 1

    def _convert_dict_of_lists_to_rows(self, data: dict) -> Iterator[dict]:
        """
        Convert a column-oriented dict (dict-of-lists) into an iterator of row dicts.
        """
        num_rows = len(next(iter(data.values())))
        keys = list(data.keys())
        for i in range(num_rows):
            yield {k: data[k][i] for k in keys}

    def _stream_json_array(self, f: io.TextIOBase) -> Iterator[Any]:
        """
        Stream JSON objects from a file containing a JSON array without loading the entire array.
        Assumes the file pointer is at the beginning of the array.
        """
        c = f.read(1)
        if c != '[':
            raise ValueError("Expected '[' at the start of a JSON array")
        decoder = json.JSONDecoder()
        buffer = ""
        while True:
            if not buffer:
                chunk = f.read(4096)
                if not chunk:
                    break
                buffer += chunk
            buffer = buffer.lstrip()
            if buffer.startswith(']'):
                break
            try:
                obj, idx = decoder.raw_decode(buffer)
                yield obj
                buffer = buffer[idx:]
                buffer = buffer.lstrip()
                if buffer.startswith(','):
                    buffer = buffer[1:]
            except json.JSONDecodeError:
                chunk = f.read(4096)
                if not chunk:
                    break
                buffer += chunk
    
    def _batch_iterable(self, iterator: Iterator[Any], batch_size: int) -> Iterator[List[Any]]:
        """
        Yield successive batches from an iterator using islice.
        """
        from itertools import islice
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    # ------------------ Public Methods ------------------ #
    
    def fetch(self, **kwargs: Any) -> Union[pd.DataFrame, list, dict]:
        """
        Fetch data from a file using the selected reader and source.

        Args:
            **kwargs: Additional arguments passed to the reader.

        Returns:
            Union[pd.DataFrame, list, dict]: Fetched data.
        """
        try:
            file_obj = self.source_handler.fetch(self.file_path, **kwargs)
            match self.file_type:
                case "csv":
                    return self._fetch_csv(file_obj, **kwargs)
                case "json":
                    return self._fetch_json(file_obj, **kwargs)
                case "parquet":
                    return self._fetch_parquet(file_obj, **kwargs)
                case "xlsx":
                    return self._fetch_excel(file_obj, **kwargs)
                case _:
                    raise ValueError(f"Unsupported file type: {self.file_type}")

        except Exception as e:
            logger.error(f"Failed to fetch data from {self.file_path}: {e}")
            raise

    def stream(self, chunk_size: int = 1024 * 1024, **kwargs: Any) -> Any:
        """Stream data from the file in chunks."""
        try:
            return self.source_handler.stream(self.file_path, chunk_size=chunk_size, **kwargs)
        except Exception as e:
            logger.error(f"Failed to stream data from {self.file_path}: {e}")
            raise

    def fetch_batch(self, batch_size: int = 1000, **kwargs: Any) -> List[Union[pd.DataFrame, list, dict]]:
        """
        Fetch data in batches.

        Args:
            batch_size (int): Number of rows per batch.
            **kwargs: Additional arguments passed to the reader.

        Returns:
            List of batches, where each batch is either a DataFrame, list, or dict
            depending on the reader and file type.

        Raises:
            ValueError: If batching is not supported for the given file type or reader.
        """
        try:
            file_obj = self.source_handler.fetch(self.file_path, **kwargs)
            
            if self.reader == "pandas":
                match self.file_type:
                    case "csv":
                        chunks = pd.read_csv(file_obj, chunksize=batch_size, **kwargs)
                        return list(chunks)
                    case "json":
                        df = pd.read_json(file_obj, lines=True, **kwargs)
                        return [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
                    case "parquet":
                        # Use pyarrow for efficient batch reading
                        table = pq.read_table(file_obj)
                        total_rows = len(table)
                        batches = []
                        
                        for start in range(0, total_rows, batch_size):
                            end = min(start + batch_size, total_rows)
                            batch_df = table.slice(start, end - start).to_pandas()
                            batches.append(batch_df)
                        
                        return batches
                    case _:
                        raise ValueError(f"Batch processing not supported for {self.file_type} files")
            
            elif self.reader == "builtin":
                # Rest of the existing builtin reader code...
                if self.file_type == "csv":
                    if isinstance(file_obj, io.BytesIO):
                        file_obj = io.TextIOWrapper(file_obj, encoding='utf-8')
                    reader = csv.DictReader(open(file_obj), **kwargs)
                    batches = []
                    current_batch = []
                    
                    for row in reader:
                        current_batch.append(row)
                        if len(current_batch) >= batch_size:
                            batches.append(current_batch)
                            current_batch = []
                    
                    if current_batch:  # Add remaining rows
                        batches.append(current_batch)
                    return batches
                
                elif self.file_type == "json":
                    # BEGIN MODIFIED: Use with statement and stream JSON objects in batches without loading entire file
                    def json_batch_generator() -> Iterator[List[Any]]:
                        with open(self.file_path, 'r', encoding='utf-8') as f:
                            pos = f.tell()
                            first_char = f.read(1)
                            while first_char and first_char.isspace():
                                first_char = f.read(1)
                            f.seek(pos)
                            
                            if first_char == '[':
                                json_iter = self._stream_json_array(f)
                            elif first_char == '{':
                                data = json.load(f)
                                if isinstance(data, dict) and self._is_dict_of_lists(data):
                                    json_iter = self._convert_dict_of_lists_to_rows(data)
                                else:
                                    json_iter = iter([data])
                            else:
                                json_iter = (json.loads(line) for line in f if line.strip())
                            
                            yield from self._batch_iterable(json_iter, batch_size)
                    
                    batch_iter = StreamingBatchIterator(json_batch_generator())
                    return batch_iter            
            else:
                raise ValueError(f"Unsupported reader: {self.reader}")

        except Exception as e:
            logger.error(f"Failed to fetch data in batches from {self.file_path}: {e}")
            raise

    def fetch_parallel(self, file_paths: List[str], **kwargs: Any) -> List[Union[pd.DataFrame, list, dict]]:
        """
        Fetch data from multiple files in parallel.
        
        For parquet files, this uses optimized parallel reading settings.
        """
        try:
            with ThreadPoolExecutor() as executor:
                if self.file_type == "parquet":
                    # Set optimized defaults for parquet
                    kwargs.setdefault('use_threads', True)
                    kwargs.setdefault('memory_map', True)
                
                futures = [executor.submit(self.fetch, file_path=path, **kwargs) for path in file_paths]
                results = [future.result() for future in as_completed(futures)]
                return results
                
        except Exception as e:
            logger.error(f"Failed to fetch data in parallel from {file_paths}: {e}")
            raise