import pytest
import io
import inspect
from typing import Iterator, List, Any
import pandas as pd

# Import the functions and classes from the utils module.
# Adjust the import path as needed.
from data_warp.connectors.utils import (
    inherit_docstring_and_signature,
    StreamingBatchIterator,
)

#############################
# Tests for the decorator   #
#############################

def dummy_func(a: int, b: int) -> int:
    """Dummy function docstring: adds two numbers."""
    return a + b

@inherit_docstring_and_signature(dummy_func)
def decorated_func(x: int, y: int) -> int:
    return x * y

def test_inherit_docstring_and_signature():
    # Verify that the decorated function inherited the docstring and signature from dummy_func.
    assert decorated_func.__doc__ == dummy_func.__doc__
    assert str(inspect.signature(decorated_func)) == str(inspect.signature(dummy_func))

#############################
# Tests for StreamingBatchIterator #
#############################

# A simple generator that yields two batches.
def simple_batch_gen() -> Iterator[List[int]]:
    yield [1, 2, 3]
    yield [4, 5, 6]

def test_iterator_next():
    it = StreamingBatchIterator(simple_batch_gen())
    # First batch using next()
    batch1 = next(it)
    assert batch1 == [1, 2, 3]
    # Second batch using next() alias.
    batch2 = it.next()
    assert batch2 == [4, 5, 6]
    with pytest.raises(StopIteration):
        next(it)

def test_to_list_and_len():
    # Create a new iterator instance.
    it = StreamingBatchIterator(simple_batch_gen())
    batches = it.to_list()
    assert batches == [[1, 2, 3], [4, 5, 6]]
    # Now, __len__ should return 2.
    assert len(it) == 2

def test_flatten_to_list():
    # Create a new iterator instance.
    it = StreamingBatchIterator(simple_batch_gen())
    flat = it.flatten_to_list()
    assert flat == [1, 2, 3, 4, 5, 6]

def test_to_dataframe():
    # Create a generator that yields batches of dictionaries.
    def dict_batch_gen() -> Iterator[List[dict]]:
        yield [{"a": 1}, {"a": 2}]
        yield [{"a": 3}]
    it = StreamingBatchIterator(dict_batch_gen())
    df = it.to_dataframe()
    expected = pd.DataFrame([{"a": 1}, {"a": 2}, {"a": 3}])
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)

def test_filter_batches():
    # Generator yields two batches: one with 3 items and one with 1 item.
    def gen() -> Iterator[List[int]]:
        yield [1, 2, 3]
        yield [4]
    it = StreamingBatchIterator(gen())
    filtered = it.filter_batches(lambda batch: len(batch) > 1)
    # Since only the first batch has more than one item:
    assert filtered.to_list() == [[1, 2, 3]]

def test_search():
    def gen() -> Iterator[List[int]]:
        yield [1, 2, 3]
        yield [4, 5, 6]
    it = StreamingBatchIterator(gen())
    # Search for even numbers
    evens = list(it.search(lambda x: x % 2 == 0))
    # Expected even numbers are 2, 4, and 6.
    assert sorted(evens) == [2, 4, 6]

def test_map_batches():
    def gen() -> Iterator[List[int]]:
        yield [1, 2, 3]
        yield [4, 5, 6]
    it = StreamingBatchIterator(gen())
    mapped = it.map_batches(lambda batch: [x * 10 for x in batch])
    mapped_list = mapped.to_list()
    assert mapped_list == [[10, 20, 30], [40, 50, 60]]

def test_flatten():
    def gen() -> Iterator[List[int]]:
        yield [1, 2]
        yield [3, 4]
    it = StreamingBatchIterator(gen())
    flat = list(it.flatten())
    assert flat == [1, 2, 3, 4]

# def test_len():
#     def gen() -> Iterator[List[int]]:
#         yield [1, 2]
#         yield [3, 4]
#     it = StreamingBatchIterator(gen())
#     # Force full evaluation.
#     _ = it.to_list()
#     assert len(it) == 2
