import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import IterableDataset
from collections import deque
from itertools import islice

"""
This module defines the `ParquetDataset` class, an iterable dataset designed for efficient, distributed
loading and processing of large Parquet files. The class supports multi-worker setups, ensuring that each 
worker processes a unique subset of data, and manages memory usage with configurable buffer sizes. 

Key Features:
- Recursively discovers and loads Parquet files from specified directories.
- Distributes data evenly across workers in a multi-GPU or distributed environment.
- Handles shuffling and epoch-based iteration with reproducibility via a user-defined seed.
- Efficiently manages large datasets by reading data in batches (buffers) and dynamically adjusting 
  buffer sizes to match available resources and dataset size.

This dataset is ideal for use in distributed training pipelines, ensuring consistent and efficient 
data access across workers while minimizing memory overhead.
"""

class ParquetDataset(IterableDataset):
    def __init__(self, paths, rank=None, world_size=None, buffer_size=20_000, seed=0, start_epoch=0, global_start_offset=0):
        super().__init__()
        
        # Initialize rank and world_size. If not explicitly provided, retrieve them from the distributed environment.
        self.rank = rank if rank is not None else dist.get_rank()
        self.world_size = world_size if world_size is not None else dist.get_world_size()
        
        self.seed = seed  # Seed for reproducibility in data shuffling.
        self.epoch = start_epoch  # Starting epoch, allowing resumption from a specific epoch.
        self.buffer_size = buffer_size  # Number of records to load into memory per batch.

        # Allow `paths` to be a string (single path) or a list of paths.
        if isinstance(paths, str):
            paths = [paths]
        
        self.files = []
        # Find all parquet files in the given paths, recursively exploring directories.
        for path in paths:
            self.files.extend(self._find_parquet_files(path))
        
        # Raise an error if no parquet files are found.
        if len(self.files) == 0:
             raise FileNotFoundError(f"No parquet files found in any of the provided paths: {paths}")
        
        # Sort the file paths for consistent ordering.
        self.files = sorted(self.files)
        # Shuffle the file list deterministically using the provided seed to ensure all ranks see the same order.
        np.random.default_rng(self.seed).shuffle(self.files)
        
        # Read the metadata for each file to determine the number of rows and compute cumulative sums.
        self.file_lengths = [pq.read_metadata(f).num_rows for f in self.files]
        self.cumulative_sums = np.cumsum(self.file_lengths)  # Cumulative sum of row counts for easy range lookups.
        self.total_length = sum(self.file_lengths)  # Total number of rows across all files.

        # Truncate total_length to multiple of world_size
        self.total_length = (self.total_length // self.world_size) * self.world_size

        # Adjust the buffer size if it exceeds the total dataset size divided across all workers.
        if self.buffer_size * self.world_size > self.total_length:
            self.buffer_size = self.total_length // self.world_size
        
        # Prevent zero or negative buffer size.
        if self.buffer_size < 1:
            raise ValueError(
                f"The dataset size is {sum(self.file_lengths)} resulting in a buffer_size of {self.buffer_size}, which is invalid. "
                "This can happen if the dataset is too small for the current world_size. "
                "Please reduce the world_size or increase the dataset size."
            )

        # The total number of records *per rank*
        self.records_per_rank = self.total_length // self.world_size
        
        # Instead of truncating, raise an error if the global_start_offset is outside [0, total_length)
        if not (0 <= global_start_offset < self.total_length):
            raise ValueError(
                f"global_start_offset ({global_start_offset}) is out of bounds for total_length ({self.total_length}). "
                "Please provide a valid offset within [0, total_length)."
            )
        
        # Calculate the per-rank offset in records, ensuring alignment with buffer boundaries.
        # The offset is distributed across ranks based on `global_start_offset // world_size`.
        # Then, align the per-rank offset to the nearest buffer boundary (lower multiple of buffer_size).
        # Example:
        #   If global_start_offset = 1000, world_size = 4, and buffer_size = 100:
        #   - global_start_offset // world_size = 250 (records per rank)
        #   - offset_per_rank = 200 (aligned to buffer boundary)
        offset_per_rank = global_start_offset // self.world_size // self.buffer_size * self.buffer_size

        # Calculate the remaining offset (in records) within the buffer after alignment.
        # Continuing the example:
        #   - offset_remaining = 250 - 200 = 50 (remaining records to skip)
        self.offset_remaining = global_start_offset // self.world_size - offset_per_rank

        # local_start and local_end define this rank's portion of the dataset
        self.local_current = (self.records_per_rank * self.rank) + offset_per_rank
        self.local_end = self.records_per_rank * (self.rank + 1)

        # Initialize an empty buffer for holding records during iteration.
        self.buffer = deque()
        
    def _find_parquet_files(self, path):
        parquet_files = []
        # Convert the input `path` to a `Path` object and resolve it to follow symbolic links,
        # ensuring we operate on the absolute path.
        initial_path = Path(path).resolve()
            
        def explore_path(p):
            # If the current path `p` is a directory, iterate over its contents.
            if p.is_dir():
                for item in p.iterdir():
                    # Recursively explore each item, resolving symbolic links to avoid issues
                    # with indirect references or loops.
                    explore_path(item.resolve())
            # If the current path `p` is a file and has a `.parquet` extension,
            # add it to the list of parquet files.
            elif p.is_file() and p.suffix == '.parquet':
                parquet_files.append(str(p))

        # Initiate the recursive exploration starting from the resolved initial path.
        explore_path(initial_path)

        # Return the list of all discovered parquet file paths.
        return parquet_files

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.total_length // self.world_size

    def __iter__(self):
        # Load the first chunk if the buffer is empty
        if len(self.buffer) == 0:
            self.load_buffer()
        
        # Yield until we run out
        while len(self.buffer) > 0:
            yield self.buffer.popleft()
            if len(self.buffer) == 0:
                self.load_buffer()  # load the next chunk

    def load_buffer(self):
        """
        Loads up to `buffer_size` records for this rank, from [local_current .. local_end).
        Returns an empty buffer if we've already read everything for the epoch.

        If the number of records read does not match the expected number, an error is raised
        to indicate a potential mismatch or corruption in the metadata. Additionally,
        handles skipping a specific number of records within the first buffer (offset_remaining)
        as part of the rank-based global offset logic.
        """
        # Start with an empty buffer
        self.buffer = deque()
        
        # 1. If we've already read all local records, end the epoch and return an empty buffer
        if self.local_current >= self.local_end:
            self.epoch += 1
            return
        
        # 2. Determine how many records remain for this rank and cap it to the buffer size
        remaining_local = self.local_end - self.local_current
        records_to_read = min(remaining_local, self.buffer_size)
        
        # Identify which file and row index correspond to the current position (local_current)
        current_file_index = np.searchsorted(
            self.cumulative_sums,
            self.local_current,
            side='right'
        )
        current_row_index = self.local_current - np.sum(self.file_lengths[:current_file_index])
        
        records_loaded = 0
                
        # 3. Read from Parquet files until we fill `records_to_read` or exhaust available files
        while records_loaded < records_to_read and current_file_index < len(self.files):
            start_idx = int(current_row_index)
            end_idx = min(
                self.file_lengths[current_file_index],
                start_idx + (records_to_read - records_loaded)
            )
        
            # Read the specified slice of rows from the current Parquet file
            df_slice = (
                pq.read_table(self.files[current_file_index])
                  .slice(start_idx, end_idx - start_idx)
                  .to_pandas()
            )
        
            # Extend the buffer with rows from the slice
            slice_count = len(df_slice)
            self.buffer.extend(df_slice.to_dict("records"))
            records_loaded += slice_count
        
            # Move to the next file if we've exhausted the rows in the current file
            if end_idx >= self.file_lengths[current_file_index]:
                current_file_index += 1
                current_row_index = 0
            else:
                current_row_index = end_idx
        
        # 4. Validate that the expected number of records was loaded
        # If there is a mismatch, raise an error to indicate potential corruption or metadata issues
        if records_loaded != records_to_read:
            raise ValueError(
                f"Expected to read {records_to_read} records, but loaded {records_loaded} instead. "
                "This suggests a mismatch in file metadata or potential data corruption."
            )
        
        # 5. Shuffle the loaded records deterministically using a rank-specific seed
        seed_val = self.seed + self.epoch + (self.local_current // self.buffer_size)
        np.random.default_rng(seed_val).shuffle(self.buffer)
        
        # 6. Skip the initial `offset_remaining` records in the first buffer
        # This aligns the rank's data processing with the distributed offset logic
        self.buffer = deque(islice(self.buffer, self.offset_remaining, None))
        self.offset_remaining = 0
                    
        # 7. Advance the local pointer by the number of records loaded in this batch
        self.local_current += records_loaded
