import functools
import inspect
from typing import Callable, Iterator, List, Any
import json

import pandas as pd



def inherit_docstring_and_signature(wrapped_func: Callable) -> Callable:
    """
    Decorator to inherit the docstring and signature of the wrapped function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(wrapped_func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        # Set the metadata on the final wrapper function
        wrapper.__doc__ = wrapped_func.__doc__
        wrapper.__signature__ = inspect.signature(wrapped_func)
        return wrapper
    return decorator

class StreamingBatchIterator:
    """
    Wraps a generator that yields batches (lists of items) and provides additional
    methods for processing the streamed data.
    """
    def __init__(self, batch_generator: Iterator[List[Any]]):
        self._gen = batch_generator  # Underlying generator yielding batches.
        self._cache = []             # Cache of batches already produced.
        self._index = 0              # Current position in the cache.
        self._exhausted = False      # Whether the underlying generator is exhausted
        
    def __iter__(self) -> 'StreamingBatchIterator':
        return self

    def __next__(self) -> List[Any]:
        # return next(self._gen)
        # If there are cached batches not yet returned, use them.
        if self._index < len(self._cache):
            result = self._cache[self._index]
            self._index += 1
            return result
        # Otherwise, if the generator is not exhausted, try to get the next batch.
        if not self._exhausted:
            try:
                result = next(self._gen)
                self._cache.append(result)
                self._index += 1
                return result
            except StopIteration:
                self._exhausted = True
        raise StopIteration

    def next(self) -> List[Any]:
        """Alias for __next__() to allow .next() calls."""
        return self.__next__()

    def to_list(self) -> List[List[Any]]:
        """
        Convert all remaining batches into a list.
        This will exhaust the iterator and update the cache.
        """
        while not self._exhausted:
            try:
                next(self)
            except StopIteration:
                break
        return self._cache
        # return list(self._gen)
    
    def flatten_to_list(self) -> List[Any]:
        """
        Convert all remaining batches into a single flat list of records.
        """
        return [record for batch in self.to_list() for record in batch]
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert all remaining batches into a single pandas DataFrame.
        This method flattens the batches into a list of records and creates a DataFrame.
        """
        records = self.flatten_to_list()
        return pd.DataFrame(records)

    def filter_batches(self, predicate: Callable[[List[Any]], bool]) -> 'StreamingBatchIterator':
        """
        Return a new StreamingBatchIterator with only batches that satisfy the predicate.
        """
        # return StreamingBatchIterator((batch for batch in self._gen if predicate(batch)))
        def filtered_gen():
            for batch in self.to_list():
                if predicate(batch):
                    yield batch
        return StreamingBatchIterator(filtered_gen())

    def search(self, search_func: Callable[[Any], bool]) -> Iterator[Any]:
        """
        Yield individual items from all batches that satisfy the search function.
        """
        for batch in self._gen:
            for item in batch:
                if search_func(item):
                    yield item

    def map_batches(self, func: Callable[[List[Any]], List[Any]]) -> 'StreamingBatchIterator':
        """
        Apply a function to each batch and return a new StreamingBatchIterator.
        """
        # return StreamingBatchIterator((func(batch) for batch in self._gen))
        def mapped_gen():
            for batch in self.to_list():
                yield func(batch)
        return StreamingBatchIterator(mapped_gen())

    def flatten(self) -> Iterator[Any]:
        """
        Flatten the batches: yield individual records from all batches.
        """
        for batch in self._gen:
            for item in batch:
                yield item

    def __len__(self) -> int:
        """
        Return the total number of batches.
        This forces full evaluation of the underlying generator.
        """
        return len(self.to_list())
        # try:
        #     remaining = self._gen.__length_hint__()
        # except AttributeError:
        #     # If not available, exhaust the generator.
        #     if not self._exhausted:
        #         self._cache.extend(list(self._gen))
        #         self._exhausted = True
        #     remaining = 0
        # return len(self._cache) + remaining