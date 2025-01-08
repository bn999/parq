import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import random
from parq.parquet_dataset import ParquetDataset

@pytest.fixture
def setup_test_dir():
    """Fixture to create and clean up the test directory."""
    test_dir = Path("test_parquet_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)

def test_parquet_dataset_multiple_ranks_with_resumption(setup_test_dir):
    """
    Test reading Parquet files across multiple ranks with proper resumption.
    """
    test_dir = setup_test_dir

    world_size = 7  # Number of ranks
    buffer_size = 10000
    num_files = 31
    start_id = 0

    # -- Create multiple Parquet files --
    for file_index in range(num_files):
        end_id = start_id + random.randint(0, 1317)
        df = pd.DataFrame({
            'id': range(start_id, end_id),
            'value': [f"val_{i}" for i in range(start_id, end_id)]
        })
        parquet_file = test_dir / f"test_data_{file_index+1}.parquet"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_file)
        start_id = end_id

    total_records = start_id
    global_offset = int(random.uniform(0.2, 0.5) * total_records)
    global_offset = global_offset // world_size * world_size

    all_data_by_rank = []

    # -- Perform partial iteration for each rank --
    for rank in range(world_size):
        dataset = ParquetDataset(
            paths=str(test_dir),
            rank=rank,
            world_size=world_size,
            start_epoch=0,
            buffer_size=buffer_size,
            global_start_offset=0
        )

        partial_data = []
        data_iter = iter(dataset)
        for _ in range(global_offset // world_size):  # Read a portion of the data
            try:
                item = next(data_iter)
                partial_data.append(item)
            except StopIteration:
                break

        # -- Create a new dataset instance for resumption --
        resumed_dataset = ParquetDataset(
            paths=str(test_dir),
            rank=rank,
            world_size=world_size,
            start_epoch=0,
            buffer_size=buffer_size,
            global_start_offset=global_offset
        )
        resumed_data = list(resumed_dataset)

        # Combine partial and resumed data
        all_data_by_rank.append(partial_data + resumed_data)

    # -- Summarize all data read across ranks --
    all_data = [item for rank_data in all_data_by_rank for item in rank_data]
    unique_ids = {x['id'] for x in all_data}

    # -- Assertions --
    assert len(all_data) == len(dataset) * world_size, "Total items read does not match the total records."
    assert len(unique_ids) == len(all_data), "Duplicate IDs detected."
