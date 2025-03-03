import io
import json
import os
import tempfile
import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import inspect
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

from data_warp.connectors.file_connector import (
    FileConnector,
    inherit_docstring_and_signature,
    StreamingBatchIterator,
)


@pytest.fixture
def csv_file():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def json_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        with open(f.name, 'w') as json_f:
            json.dump(data, json_f)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def json_file_dict_of_lists():
    # This JSON file is a dict-of-lists.
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(data, f)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def json_file_array():
    # This JSON file is a JSON array.
    data = [{"a": 1}, {"a": 2}]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(data, f)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def json_file_ndjson():
    # This JSON file is NDJSON.
    data = [{"x": "one"}, {"x": "two"}, {"x": "three"}]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        for rec in data:
            f.write(json.dumps(rec) + "\n")
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def parquet_file():
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, f.name)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def excel_file():
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        df.to_excel(f.name, index=False)
        yield f.name
    os.unlink(f.name)

# ----------------------- Basic Fetch Tests ----------------------- #

# ----------------------- CSV Tests ----------------------- #

def test_fetch_csv(csv_file):
    connector = FileConnector(file_path=csv_file, source="local")
    data = connector.fetch()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (2, 2)
    assert data.iloc[0]['name'] == 'Alice'

def test_fetch_csv_with_builtin(csv_file):
    connector = FileConnector(file_path=csv_file, source="local", reader="builtin")
    data = connector.fetch()
    print("Builtin CSV Data: ", data)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]['name'] == 'Alice'

def test_fetch_csv_pyarrow(csv_file):
    connector = FileConnector(file_path=csv_file, source="local", reader="pyarrow")
    # Patch pyarrow's read_csv to simulate a DataFrame result.
    with patch("data_warp.connectors.file_connector.pacsv.read_csv") as mock_pyarrow_read_csv:
        df_expected = pd.read_csv(csv_file)
        mock_pyarrow_read_csv.return_value = df_expected
        data = connector.fetch()
        mock_pyarrow_read_csv.assert_called_once()
        assert data.equals(df_expected)

# ----------------------- JSON Tests ----------------------- #

def test_fetch_json(json_file):
    connector = FileConnector(file_path=json_file, file_type="json", source="local")
    data = connector.fetch()
    assert isinstance(data, pd.DataFrame)
    assert list(data['name']) == ["Alice", "Bob"]

def test_fetch_json_builtin(json_file):
    connector = FileConnector(file_path=json_file, file_type="json", source="local", reader="builtin")
    data = connector.fetch()
    # For non-BytesIO file, builtin returns the entire object (a dict in this case)
    assert isinstance(data, dict)
    assert data["name"] == ["Alice", "Bob"]

def test_fetch_json_pyarrow(json_file):
    connector = FileConnector(file_path=json_file, file_type="json", source="local", reader="pyarrow")
    with patch("data_warp.connectors.file_connector.pajson.read_json") as mock_pyarrow_read_json:
        df_expected = pd.read_json(json_file)
        mock_pyarrow_read_json.return_value = df_expected
        data = connector.fetch()
        mock_pyarrow_read_json.assert_called_once()
        assert data.equals(df_expected)

#----------------------- Parquet Tests ----------------------- #

def test_fetch_parquet(parquet_file):
    connector = FileConnector(file_path=parquet_file, file_type="parquet", source="local")
    data = connector.fetch()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (2, 2)
    assert data.iloc[0]['name'] == 'Alice'

def test_fetch_parquet_pyarrow(parquet_file):
    connector = FileConnector(file_path=parquet_file, file_type="parquet", source="local", reader="pyarrow")
    with patch("data_warp.connectors.file_connector.pq.read_table") as mock_pq_read_table:
        df_expected = pd.read_parquet(parquet_file)
        # Simulate a pyarrow Table whose .to_pandas() returns our DataFrame.
        dummy_table = MagicMock()
        dummy_table.to_pandas.return_value = df_expected
        mock_pq_read_table.return_value = dummy_table
        data = connector.fetch()
        #mock_pq_read_table.assert_called_once()
        assert data.equals(df_expected)

# ----------------------- Excel Tests ----------------------- #

def test_fetch_excel(excel_file):
    connector = FileConnector(file_path=excel_file, file_type="xlsx", source="local")
    data = connector.fetch()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (2, 2)
    assert data.iloc[0]['name'] == 'Alice'

def test_fetch_excel_with_invalid_reader(excel_file):
    connector = FileConnector(file_path=excel_file, file_type="xlsx", reader="builtin")
    with pytest.raises(ValueError):
        connector.fetch()

#########################
# Tests for Batch Fetching
#########################


def test_fetch_batch_csv_builtin(csv_file):
    connector = FileConnector(file_path=csv_file, source="local", reader="builtin")
    batches = connector.fetch_batch(batch_size=1)
    # Builtin CSV batching should produce 2 batches (one for each row)
    if isinstance(batches, StreamingBatchIterator):
        batch_list = batches.flatten_to_list()
    else:
        batch_list = batches
    assert isinstance(batch_list, list)
    assert len(batch_list) == 2
    total_rows = sum(len(batch) for batch in batch_list)
    assert total_rows == 2
    assert batch_list[0][0]['name'] == 'Alice'

def test_fetch_batch_json_builtin_array(json_file_array):
    # Test builtin JSON batching for a JSON array.
    connector = FileConnector(file_path=str(json_file_array), file_type="json", source="local", reader="builtin")
    batches = connector.fetch_batch(batch_size=1)
    batche_df = batches.to_dataframe()
    # Expect 2 batches: one per object in the array.
    assert isinstance(batche_df, pd.DataFrame)


def test_fetch_batch_json_builtin_dict_of_lists(json_file_dict_of_lists):
    # Test builtin JSON batching for a dict-of-lists.
    connector = FileConnector(file_path=str(json_file_dict_of_lists), file_type="json", source="local", reader="builtin")
    batches = connector.fetch_batch(batch_size=1)
    batche_df = batches.to_dataframe()
    # Expect 2 batches: one per object in the array.
    assert isinstance(batche_df, pd.DataFrame)

def test_fetch_batch_json_builtin_ndjson(json_file_ndjson):
    # Test builtin JSON batching for NDJSON.
    connector = FileConnector(file_path=str(json_file_ndjson), file_type="json", source="local", reader="builtin")
    batches = connector.fetch_batch(batch_size=2)

    batche_df = batches.to_dataframe()
    # Expect 2 batches: one per object in the array.
    assert isinstance(batche_df, pd.DataFrame)

def test_fetch_batch_parquet_chunked(parquet_file):
    connector = FileConnector(file_path=parquet_file, file_type="parquet", source="local", chunk_size=1)
    batches = connector.fetch_batch(batch_size=1)
    # Expect a list of DataFrame batches.
    assert isinstance(batches, list)
    first_batch = batches[0]
    assert isinstance(first_batch, pd.DataFrame)
    assert first_batch.shape[0] >= 1

# ----------------------- Error Handling Tests: Unsupported ----------------------- #

def test_fetch_unsupported_file_type(csv_file):
    connector = FileConnector(file_path=csv_file, file_type="unsupported", source="local")
    with pytest.raises(ValueError, match="Unsupported file type: unsupported"):
        connector.fetch()

def test_fetch_csv_with_unsupported_reader(csv_file):
    connector = FileConnector(file_path=csv_file, reader="unsupported", source="local")
    with pytest.raises(ValueError, match="Unsupported reader for CSV: unsupported"):
        connector.fetch()

def test_fetch_file_not_found():
    connector = FileConnector(file_path="nonexistent.csv", source="local")
    with pytest.raises(FileNotFoundError):
        connector.fetch()
    
# ----------------------- HTTP & S3 Source Tests ----------------------- #

@patch("requests.get")
def test_fetch_http_file_source(mock_get):
    mock_response = MagicMock()
    mock_response.content = b"name,age\nAlice,25\nBob,30"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    connector = FileConnector(file_path="http://example.com/data.csv", source="http")
    data = connector.fetch()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (2, 2)
    assert data.iloc[0]['name'] == 'Alice'

@patch("requests.get")
def test_fetch_http_file_source_with_retry(mock_get):
    mock_get.side_effect = [requests.RequestException("Failed"), MagicMock(content=b"name,age\nAlice,25\nBob,30")]

    connector = FileConnector(file_path="http://example.com/data.csv", source="http")
    data = connector.fetch()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (2, 2)
    assert data.iloc[0]['name'] == 'Alice'

@patch("boto3.client")
def test_fetch_s3_file_source(mock_boto3):
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": io.BytesIO(b"name,age\nAlice,25\nBob,30")}
    mock_boto3.return_value = mock_s3

    connector = FileConnector(file_path="my-bucket/data.csv", source="s3")
    data = connector.fetch()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (2, 2)
    assert data.iloc[0]['name'] == 'Alice'

# ----------------------- Streaming Tests ----------------------- #

def test_stream_csv(csv_file):
    connector = FileConnector(file_path=csv_file, source="local", streaming=True)
    stream = connector.stream()
    assert isinstance(stream, Iterator)
    for chunk in stream:
        assert isinstance(chunk, bytes)

def test_stream_json(json_file):
    connector = FileConnector(file_path=json_file, file_type="json", source="local", streaming=True)
    stream = connector.stream()
    assert isinstance(stream, Iterator)
    for chunk in stream:
        assert isinstance(chunk, bytes)

def test_stream_parquet(parquet_file):
    connector = FileConnector(file_path=parquet_file, file_type="parquet", source="local", streaming=True)
    stream = connector.stream()
    assert isinstance(stream, Iterator)
    for chunk in stream:
        assert isinstance(chunk, bytes)

def test_stream_excel(excel_file):
    connector = FileConnector(file_path=excel_file, file_type="excel", source="local", streaming=True)
    stream = connector.stream()
    assert isinstance(stream, Iterator)
    for chunk in stream:
        assert isinstance(chunk, bytes)

# ----------------------- Batch Fetching Tests ----------------------- #

def test_fetch_csv_chunked(csv_file):
    connector = FileConnector(file_path=csv_file, source="local", chunk_size=1)
    data = connector.fetch_batch()
    print("Data: ", data)
    dataDF = data[0]
    assert isinstance(data, list)
    assert isinstance(dataDF, pd.DataFrame)
    assert dataDF.shape == (2, 2)
    assert dataDF.iloc[0]['name'] == 'Alice'

def test_fetch_json_chunked(json_file):
    connector = FileConnector(file_path=json_file, file_type="json", source="local", chunk_size=1)
    data = connector.fetch_batch()
    dataDF = data[0]
    assert isinstance(dataDF, pd.DataFrame)
    assert dataDF.iloc[0]['name'] == ['Alice', 'Bob']

def test_fetch_parquet_chunked(parquet_file):
    connector = FileConnector(file_path=parquet_file, file_type="parquet", source="local", chunk_size=1)
    data = connector.fetch_batch()
    dataDF = data[0]
    assert isinstance(dataDF, pd.DataFrame)
    assert dataDF.shape == (2, 2)
    assert dataDF.iloc[0]['name'] == 'Alice'

# def test_fetch_excel_chunked(excel_file):
#     connector = FileConnector(file_path=excel_file, file_type="xlsx", source="local", chunk_size=1)
#     data = connector.fetch_batch()
#     assert isinstance(data, pd.DataFrame)
#     assert data.shape == (2, 2)
#     assert data.iloc[0]['name'] == 'Alice'

def test_fetch_batch_csv_builtin(csv_file):
    connector = FileConnector(file_path=csv_file, source="local", reader="builtin")
    batches = connector.fetch_batch(batch_size=1)
    # builtin CSV batching returns a list of lists (each inner list is a batch of dicts)
    assert isinstance(batches, list)
    print("Batches: ", batches)

    total_rows = sum(len(batch) for batch in batches)
    print("Total Rows: ", total_rows)
    assert total_rows == 2
    assert batches[0][0]['name'] == 'Alice'

def test_fetch_batch_json_builtin(json_file):
    connector = FileConnector(file_path=json_file, file_type="json", source="local", reader="builtin")
    batches_gen = connector.fetch_batch()
    batches = batches_gen.flatten_to_list()
    # For JSON builtin, if the loaded data is not a list, it returns [data]
    assert isinstance(batches, list)
    # Our fixture JSON is a dict so we expect a single batch.
    assert len(batches) == 2

def test_fetch_batch_with_unsupported_reader(csv_file):
    # For fetch_batch, if reader is not 'pandas' or 'builtin', it should raise a ValueError.
    connector = FileConnector(file_path=csv_file, source="local", reader="pyarrow")
    with pytest.raises(ValueError, match="Unsupported reader: pyarrow"):
        connector.fetch_batch()



# ----------------------- Error Propagation in Public Methods ----------------------- #

def test_stream_error(csv_file):
    connector = FileConnector(file_path=csv_file, source="local", streaming=True)
    # Patch the source_handler's stream method to raise an exception.
    connector.source_handler.stream = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("stream error"))
    with pytest.raises(Exception, match="stream error"):
        list(connector.stream())

def test_fetch_batch_error(csv_file):
    connector = FileConnector(file_path=csv_file, source="local", reader="pandas")
    # Patch source_handler.fetch to raise an exception.
    connector.source_handler.fetch = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("fetch error"))
    with pytest.raises(Exception, match="fetch error"):
        connector.fetch_batch(batch_size=1)

# ----------------------- Invalid Source Tests ----------------------- #

def test_fetch_invalid_source():
    with pytest.raises(ValueError, match="Unsupported file source: invalid"):
        FileConnector(file_path="test.csv", source="invalid")

def test_stream_invalid_source():
    with pytest.raises(ValueError, match="Unsupported file source: invalid"):
        FileConnector(file_path="test.csv", source="invalid", streaming=True)

def test_fetch_csv_with_invalid_reader(csv_file):
    connector = FileConnector(file_path=csv_file, reader="invalid", source="local")
    with pytest.raises(ValueError, match="Unsupported reader for CSV: invalid"):
        connector.fetch()

def test_fetch_invalid_file_type(csv_file):
    connector = FileConnector(file_path=csv_file, file_type="invalid", source="local")
    with pytest.raises(ValueError, match="Unsupported file type: invalid"):
        connector.fetch()

# ----------------------- Decorator Tests ----------------------- #
def test_inherit_docstring_and_signature(csv_file):
    connector = FileConnector(file_path=csv_file, source="local")
    # Check that the decorated _fetch_csv_pandas has inherited doc and signature from pd.read_csv
    expected_doc = pd.read_csv.__doc__
    expected_sig = str(inspect.signature(pd.read_csv))
    assert connector._fetch_csv_pandas.__doc__ == expected_doc
    assert str(connector._fetch_csv_pandas.__signature__) == expected_sig

# ----------------------- _fetch_json_builtin BytesIO Test ----------------------- #
def test_fetch_json_builtin_bytes_io(json_file):
    with open(json_file, "rb") as f:
        bio = io.BytesIO(f.read())
    connector = FileConnector(file_path=json_file, file_type="json", source="local", reader="builtin")
    # Patch source_handler.fetch to return the BytesIO
    connector.source_handler.fetch = lambda path, **kwargs: bio
    data = connector._fetch_json_builtin(bio)
    assert isinstance(data, dict)
    assert data["name"] == ["Alice", "Bob"]

# ----------------------- _infer_file_type Test ----------------------- #
def test_infer_file_type():
    connector = FileConnector(file_path="test.JSON", source="local")
    assert connector._infer_file_type("test.JSON") == "json"