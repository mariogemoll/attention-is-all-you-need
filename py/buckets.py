import os
import struct
from contextlib import contextmanager
from functools import partial
from multiprocessing import Pool
from typing import BinaryIO, Generator

from dataset import Dataset, get_entry, open_dataset


class BucketedDataset:
    def __init__(
        self, bucket_index_file: BinaryIO, step_size: int, num_buckets: int, dataset: Dataset
    ):
        self.bucket_index_file = bucket_index_file
        self.step_size = step_size
        self.num_buckets = num_buckets
        self.dataset = dataset

    def iter_bucket_entries(
        self, bucket_id: int
    ) -> Generator[tuple[int, int, int, int, list[int], list[int]], None, None]:
        """
        Iterate over entries in a specific bucket.

        Args:
            bucket_id: ID of the bucket to iterate over

        Yields:
            Tuple of (bucket_id, idx_in_bucket, corpus_id, original_line_number, src_tokens,
            tgt_tokens)
        """
        bucket_sizes = get_bucket_sizes(self.bucket_index_file)

        if bucket_id >= len(bucket_sizes):
            return

        bucket_size = bucket_sizes[bucket_id]

        for idx_in_bucket in range(bucket_size):
            entry_idx = get_entry_idx_from_bucket(self.bucket_index_file, bucket_id, idx_in_bucket)
            corpus_id, original_line_number, src_tokens, tgt_tokens = get_entry(
                self.dataset, entry_idx
            )
            yield bucket_id, idx_in_bucket, corpus_id, original_line_number, src_tokens, tgt_tokens

    def iter_all_entries(
        self,
    ) -> Generator[tuple[int, int, int, int, list[int], list[int]], None, None]:
        """
        Iterate over all entries in all buckets.

        Yields:
            Tuple of (bucket_id, idx_in_bucket, corpus_id, original_line_number, src_tokens,
            tgt_tokens)
        """
        for bucket_id in range(self.num_buckets):
            yield from self.iter_bucket_entries(bucket_id)


@contextmanager
def open_buckets(base_path: str) -> Generator[BucketedDataset, None, None]:
    bucket_index_file = open(f"{base_path}.bidx", "rb")
    try:
        step_size, num_buckets, _ = read_bucket_index_header(bucket_index_file)
        with open_dataset(base_path) as dataset:
            yield BucketedDataset(
                bucket_index_file=bucket_index_file,
                step_size=step_size,
                num_buckets=num_buckets,
                dataset=dataset,
            )
    finally:
        bucket_index_file.close()


def _process_index_chunk(
    src_idx_path: str,
    tgt_idx_path: str,
    chunk_start: int,
    chunk_size: int,
    entries_count: int,
    step_size: int,
    num_buckets: int,
) -> list[list[int]]:
    """
    Process a chunk of entries by reading only the assigned range.

    Args:
        src_idx_path: Path to the src.idx file
        tgt_idx_path: Path to the tgt.idx file
        chunk_start: Starting entry index for this chunk
        chunk_size: Number of entries to process in this chunk
        entries_count: Total number of entries in the dataset
        step_size: Step size for buckets
        num_buckets: Total number of buckets

    Returns:
        List of lists, where buckets[i] contains entry indices for bucket i
    """
    buckets: list[list[int]] = [[] for _ in range(num_buckets)]
    chunk_end = min(chunk_start + chunk_size, entries_count)

    with open(src_idx_path, "rb") as src_idx, open(tgt_idx_path, "rb") as tgt_idx:
        for entry_idx in range(chunk_start, chunk_end):
            # Get src token count
            src_idx.seek(entry_idx * 4)
            src_end = struct.unpack("<I", src_idx.read(4))[0]
            src_start = 0
            if entry_idx > 0:
                src_idx.seek((entry_idx - 1) * 4)
                src_start = struct.unpack("<I", src_idx.read(4))[0]
            src_len = (src_end - src_start) // 2

            # Get tgt token count
            tgt_idx.seek(entry_idx * 4)
            tgt_end = struct.unpack("<I", tgt_idx.read(4))[0]
            tgt_start = 0
            if entry_idx > 0:
                tgt_idx.seek((entry_idx - 1) * 4)
                tgt_start = struct.unpack("<I", tgt_idx.read(4))[0]
            tgt_len = (tgt_end - tgt_start) // 2

            effective_tgt_len = tgt_len + 1  # Add 1 for EOS token
            max_len = max(src_len, effective_tgt_len)

            bucket_idx = min((max_len + step_size - 1) // step_size - 1, num_buckets - 1)
            if bucket_idx < 0:
                bucket_idx = 0

            buckets[bucket_idx].append(entry_idx)

    return buckets


def create_bucket_index(
    dataset_file_path_prefix: str,
    step_size: int,
    max_length: int,
    num_processes: int,
) -> int:
    """
    Create a bucket index file that groups entry indices by maximum sentence length.

    The bucket index format:
    - Header: [step_size(4B)] [num_buckets(4B)] [bucket_offset_1(4B)] [bucket_offset_2(4B)] ...
    - For each bucket: entry indices with format [entry_idx(4B)] = 4 bytes per entry

    Args:
        dataset_file_path: Path to the dataset file (without suffix)
        step_size: Step size for buckets (e.g., 16 means buckets: 16, 32, 48...)
        max_length: Maximum length to consider
        num_processes: Number of processes to use (None for CPU count)

    Returns:
        Total number of entries processed
    """
    src_idx_path = dataset_file_path_prefix + ".src.idx"
    tgt_idx_path = dataset_file_path_prefix + ".tgt.idx"
    output_index_path = dataset_file_path_prefix + ".bidx"
    if not os.path.exists(src_idx_path):
        raise FileNotFoundError(f"Dataset src.idx file not found: {src_idx_path}")
    if not os.path.exists(tgt_idx_path):
        raise FileNotFoundError(f"Dataset tgt.idx file not found: {tgt_idx_path}")
    src_idx_size = os.path.getsize(src_idx_path)
    num_entries = src_idx_size // 4
    num_buckets = (max_length + step_size - 1) // step_size

    chunk_size = max(1, num_entries // num_processes)
    chunks = []
    for i in range(0, num_entries, chunk_size):
        remaining = num_entries - i
        actual_chunk_size = min(chunk_size, remaining)
        chunks.append((i, actual_chunk_size))

    # Process chunks in parallel
    worker_func = partial(
        _process_index_chunk,
        src_idx_path,
        tgt_idx_path,
        entries_count=num_entries,
        step_size=step_size,
        num_buckets=num_buckets,
    )
    with Pool(processes=num_processes) as pool:
        results = list(pool.starmap(worker_func, chunks))

    # Merge results from all processes
    buckets: list[list[int]] = [[] for _ in range(num_buckets)]
    total_entries = 0
    for worker_buckets in results:
        for bucket_idx, entries in enumerate(worker_buckets):
            buckets[bucket_idx].extend(entries)
            total_entries += len(entries)

    # Write bucket index file
    with open(output_index_path, "wb") as output_file:
        # Write header: step size, num. of buckets, followed by bucket offsets (to be filled later)
        output_file.write(struct.pack("<I", step_size))
        output_file.write(struct.pack("<I", num_buckets))

        # Reserve space for bucket offsets (will fill in later)
        bucket_offset_positions = []
        for _ in range(num_buckets):
            bucket_offset_positions.append(output_file.tell())
            output_file.write(struct.pack("<I", 0))  # Placeholder

        # Write bucket data and record actual offsets
        bucket_offsets = []
        entries_per_bucket = []

        for bucket_idx, bucket_entries in enumerate(buckets):
            bucket_start_offset = output_file.tell()
            bucket_offsets.append(bucket_start_offset)
            entries_per_bucket.append(len(bucket_entries))

            # Write all entry indices in this bucket
            for entry_idx in bucket_entries:
                output_file.write(struct.pack("<I", entry_idx))

        # Go back and fill in the actual bucket offsets
        for i, (offset_pos, actual_offset) in enumerate(
            zip(bucket_offset_positions, bucket_offsets)
        ):
            output_file.seek(offset_pos)
            output_file.write(struct.pack("<I", actual_offset))

    return total_entries


def read_bucket_index_header(
    bucket_index_file: BinaryIO,
) -> tuple[int, int, list[int]]:
    """
    Read the header of a bucket index file to get bucket information.

    Args:
        bucket_index_file: Open binary file handle for reading bucket index

    Returns:
        Tuple of (step_size, num_buckets, bucket_offsets)
    """
    bucket_index_file.seek(0)
    step_size = struct.unpack("<I", bucket_index_file.read(4))[0]
    num_buckets = struct.unpack("<I", bucket_index_file.read(4))[0]
    bucket_offsets = []

    for _ in range(num_buckets):
        offset = struct.unpack("<I", bucket_index_file.read(4))[0]
        bucket_offsets.append(offset)

    return step_size, num_buckets, bucket_offsets


def get_bucket_sizes(
    bucket_index_file: BinaryIO,
) -> list[int]:
    """
    Get the number of entries in each bucket from an opened bucket index file.

    Args:
        bucket_index_file: Open binary file handle for reading bucket index

    Returns:
        List of entry counts for each bucket
    """
    step_size, num_buckets, bucket_offsets = read_bucket_index_header(bucket_index_file)

    bucket_sizes = []
    # Save current position
    current_pos = bucket_index_file.tell()
    # Get file size robustly
    bucket_index_file.seek(0, 2)
    file_size = bucket_index_file.tell()
    bucket_index_file.seek(current_pos)
    for i in range(num_buckets):
        start_offset = bucket_offsets[i]
        end_offset = bucket_offsets[i + 1] if i + 1 < num_buckets else file_size
        bucket_byte_size = end_offset - start_offset
        entry_count = bucket_byte_size // 4
        bucket_sizes.append(entry_count)
    return bucket_sizes


def get_bucket_size(
    bucket_index_file: BinaryIO,
    bucket_id: int,
) -> int:
    """
    Get the number of entries in a specific bucket from an opened bucket index file.

    Args:
        bucket_index_file: Open binary file handle for reading bucket index
        bucket_id: ID of the bucket to get size for

    Returns:
        Number of entries in the specified bucket
    """
    bucket_sizes = get_bucket_sizes(bucket_index_file)
    if bucket_id >= len(bucket_sizes):
        raise ValueError(f"Bucket ID {bucket_id} >= number of buckets {len(bucket_sizes)}")
    return bucket_sizes[bucket_id]


def get_entry_idx_from_bucket(
    bucket_index_file: BinaryIO,
    bucket_id: int,
    idx_in_bucket: int,
) -> int:
    """
    Get the entry index for a specific position within a bucket from a bucket index file.

    Args:
        bucket_index_file: Open binary file handle for reading bucket index
        bucket_id: ID of the bucket to read from
        idx_in_bucket: Index within the bucket (0-based)

    Returns:
        The entry index from the original dataset

    Raises:
        ValueError: If bucket_id is invalid or idx_in_bucket is out of range
    """
    # Read header to get bucket information
    bucket_index_file.seek(0)
    _ = struct.unpack("<I", bucket_index_file.read(4))[0]  # Step size
    num_buckets = struct.unpack("<I", bucket_index_file.read(4))[0]

    if bucket_id >= num_buckets:
        raise ValueError(f"Bucket ID {bucket_id} >= number of buckets {num_buckets}")

    # Read the specific bucket offset we need
    # Skip step_size + num_buckets + previous bucket offsets
    bucket_index_file.seek(8 + bucket_id * 4)
    bucket_start_offset = struct.unpack("<I", bucket_index_file.read(4))[0]

    # Jump directly to the entry within the bucket
    entry_offset = bucket_start_offset + (idx_in_bucket * 4)
    bucket_index_file.seek(entry_offset)

    # Read and return the entry index
    entry_data = bucket_index_file.read(4)
    if len(entry_data) < 4:
        raise ValueError(f"Could not read entry at bucket {bucket_id}, index {idx_in_bucket}")

    entry_idx: int = struct.unpack("<I", entry_data)[0]
    return entry_idx
