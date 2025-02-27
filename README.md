# Data Buddy

`data-buddy` is a Python library for building scalable data workflows with support for multiple backends like Pandas, Spark, and Flink. It simplifies data pipelines, automates dependency setup, and works seamlessly across different environments, including Docker.
 
---

## Features (Work In progress. Following features list will change)

- **FileConnector**: Connector for different formats like csv, json, parquet.
- **Abstract Pipeline Design**: Easily build and extend workflows using `BasePipeline`.
- **Spark Batch Support**: Use the `SparkBatchPipeline` for scalable, distributed data processing.
- **Automated Dependency Management**: Automatically installs Java and sets up `JAVA_HOME` for Spark compatibility.
- **Cross-Platform**: Works on Linux, macOS, Windows, and Docker with minimal setup.

---

## Installation 

### Basic Installation
Install `data-buddy` with Pandas support:
```bash
pip install data-buddy
