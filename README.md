[![codecov](https://codecov.io/gh/dattatele/data-warp/branch/master/graph/badge.svg)](https://codecov.io/gh/dattatele/data-warp) [![PyPI](https://img.shields.io/pypi/v/data-warp.svg)](https://pypi.org/project/data-warp/) 

# Data-Warp 🌀
  
`Data-Warp` is a powerful Python library that simplifies working with data files across various formats and storage locations. At its core is the FileConnector module, a universal connector designed to streamline data ingestion from multiple sources with minimal configuration.

A one-stop-shop for all data operations, connectors, orchestration, transformation, ELT, monitoring, dashboards, and reporting for data engineers. 

![Simplification-2](https://github.com/user-attachments/assets/ba9bf6b9-6ef9-4c54-9f25-a4e8415fb1bf)


## Key Features

- Multiple File Formats: Native support for CSV, JSON, Parquet, Excel, and extensible to other formats
- Diverse Data Sources: Connect to files from local storage, HTTP endpoints, AWS S3, and more
- Flexible Reading Engines: Choose between pandas, Python built-ins, or PyArrow for optimal performance
- Efficient Data Handling:
      - Streaming capability for memory-efficient processing of large files
      - Batch processing to handle data in manageable chunks
      - Parallel data fetching for improved performance
- Error Handling: Built-in retry mechanisms and comprehensive error reporting
- User-Friendly API: Simple, consistent interface regardless of underlying data source or format

## Use Cases

- Data engineering pipelines requiring connection to various data sources
- ETL processes working with multiple file formats
- Data science workflows needing efficient data loading
- Applications requiring streaming capabilities for large datasets
- Cross-platform data access with consistent API
  
## Example      
 
```
# Connect to a local CSV file
connector = FileConnector(file_path="data.csv", source="local")
data = connector.fetch()

# Stream a large JSON file from S3
s3_connector = FileConnector(
    file_path="s3://bucket/large_data.json",
    file_type="json",
    source="s3",
    streaming=True
)
for chunk in s3_connector.stream(chunk_size=10000):
    process_data(chunk)

# Fetch multiple files in parallel
connector = FileConnector(file_path="data.parquet", source="local")
results = connector.fetch_parallel(["file1.parquet", "file2.parquet", "file3.parquet"]) 

# Fetch files in batch with builtin support with additional supportive methods,

# fetch_batch #built-in file format
FileConnector("huge_csv_file.csv", reader="builtin").fetch_batch()


# For json has various additional methods to deal with large files and useful for ad-hoc filter

#search
list(FileConnector(huge_json_file.json, reader="builtin").fetch_batch().search(lambda rec: rec.get("hash_id")=="6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b"))

#filter
filtered = FileConnector(huge_json_file.json, reader="builtin").fetch_batch().filter_batches(lambda rec: rec[0].get("int_field") < 8516)
print("filtered", filtered.next())

# Map batches:
mapped =  FileConnector(huge_json_file.json, reader="builtin").fetch_batch().map_batches(
    lambda batch: [rec for rec in batch if rec.get("date") < "2002-07-06"]
)
for batch in mapped:
    print("Mapped batch:", batch)

FileConnector(huge_json_file.json, reader="builtin").fetch_batch().to_dataframe().head()
     
```

## Installation 

### Basic Installation

```bash
pip install data-warp --update
```
 
