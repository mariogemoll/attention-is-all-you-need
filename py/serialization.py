import os
import random
import struct
from functools import partial
from multiprocessing import Pool
from typing import BinaryIO

from tqdm import tqdm, trange


def append_to_dataset(
    data_file: BinaryIO,
    index_file: BinaryIO,
    corpus_id: int,
    original_line_number: int,
    src_tokens: list[int],
    tgt_tokens: list[int],
) -> None:
    """
    Append a single entry to the dataset files.

    Data format: [corpus_id(1B)] [line_number(4B)] [src_tokens(2B each)] [tgt_tokens(2B each)]
    Index format: [entry_start_pos(4B)] [src_token_count(1B)] = 5 bytes per entry

    Args:
        data_file: Open binary file handle for writing data
        index_file: Open binary file handle for writing index
        corpus_id: ID of the corpus (1 byte unsigned int)
        original_line_number: Original line number in source file
        src_tokens: List of source token IDs (max 255 tokens)
        tgt_tokens: List of target token IDs
    """
    # Get current position for index
    entry_start_pos = data_file.tell()

    # Validate source token count fits in a byte
    src_token_count = len(src_tokens)
    if src_token_count > 255:
        raise ValueError(f"Source token count {src_token_count} exceeds maximum of 255")

    # Write corpus_id (1 byte unsigned int)
    data_file.write(struct.pack("<B", corpus_id))

    # Write original line number (32-bit little-endian unsigned int)
    data_file.write(struct.pack("<I", original_line_number))

    # Write first sentence tokens (batch pack for efficiency)
    if src_tokens:
        src_packed = struct.pack("<" + "H" * len(src_tokens), *src_tokens)
        data_file.write(src_packed)

    # Write second sentence tokens (batch pack for efficiency)
    if tgt_tokens:
        tgt_packed = struct.pack("<" + "H" * len(tgt_tokens), *tgt_tokens)
        data_file.write(tgt_packed)

    # Write index entry (entry start position + source token count)
    # Format: 32-bit entry position + 8-bit source token count = 5 bytes total
    index_file.write(struct.pack("<IB", entry_start_pos, src_token_count))


def combine_datasets(
    output_file_path: str,
    input_file_paths: list[str],
) -> int:
    """
    Combine multiple binary dataset files into a single output file.

    Args:
        output_file_path: Path where the combined dataset will be written
        input_file_paths: List of paths to binary dataset files to combine
        cleanup_input_files: If True, delete input files after combining
        desc: Description for progress bar

    Returns:
        Total number of entries in the combined dataset
    """
    data_file_path = output_file_path + ".bin"
    index_file_path = output_file_path + ".idx"

    # Prepare list of (data_file, index_file, entry_count) tuples
    input_files = []
    total_entries = 0

    for input_path in input_file_paths:
        input_data_path = input_path + ".bin"
        input_index_path = input_path + ".idx"

        if not os.path.exists(input_data_path) or not os.path.exists(input_index_path):
            print(f"Warning: Skipping {input_path} (missing data or index file)")
            continue

        # Calculate number of entries from index file size
        index_size = os.path.getsize(input_index_path)
        entry_count = index_size // 5  # Each index entry is now 5 bytes (4 + 1)

        input_files.append((input_data_path, input_index_path, entry_count))
        total_entries += entry_count

    if not input_files:
        return 0

    print(f"Total entries to combine: {total_entries}")

    with open(data_file_path, "wb") as final_data_file, open(
        index_file_path, "wb"
    ) as final_index_file:
        current_data_offset = 0

        for input_data_path, input_index_path, entry_count in tqdm(
            input_files, desc="Combining files"
        ):
            if entry_count == 0:
                continue

            # Stream copy data file content in chunks to avoid loading entire file into memory
            data_size = 0
            with open(input_data_path, "rb") as input_data:
                while True:
                    chunk = input_data.read(64 * 1024)  # 64KB chunks
                    if not chunk:
                        break
                    final_data_file.write(chunk)
                    data_size += len(chunk)

            # Read index file and update offsets
            with open(input_index_path, "rb") as input_index:
                while True:
                    index_entry = input_index.read(5)  # 32-bit int + 8-bit byte
                    if not index_entry:
                        break

                    # Unpack the original entry position and source token count
                    entry_pos, src_token_count = struct.unpack("<IB", index_entry)

                    # Update entry position with current data offset
                    updated_entry_pos = entry_pos + current_data_offset

                    # Write updated index entry
                    final_index_file.write(struct.pack("<IB", updated_entry_pos, src_token_count))

            # Update offset for next file
            current_data_offset += data_size

    print(
        f"Combined {len(input_files)} files ({total_entries} entries) into "
        f"{output_file_path}.bin (+.idx)"
    )

    return total_entries


def get_entry_info_from_index(
    index_file: BinaryIO,
    data_file_size: int,
    entry_idx: int,
) -> tuple[int, int, int]:
    """
    Get entry position and token lengths for a specific entry from index file only.

    Args:
        index_file: Open binary file handle for reading index
        data_file_size: Size of the data file in bytes (for calculating last entry length)
        entry_idx: Index of the entry to read

    Returns:
        Tuple of (entry_start_pos, src_token_count, tgt_token_count)
    """
    # Read index entry to get entry position and source token count
    index_file.seek(entry_idx * 5)  # Each index entry is 5 bytes
    index_entry = index_file.read(5)
    if len(index_entry) < 5:
        raise ValueError(f"Invalid entry index {entry_idx}")

    entry_start_pos, src_token_count = struct.unpack("<IB", index_entry)

    # Calculate target token count by finding where this entry ends
    # Try to read next index entry to get end position
    index_file.seek((entry_idx + 1) * 5)
    next_index_entry = index_file.read(5)

    if len(next_index_entry) == 5:
        # Next entry exists - use its start position as our end position
        next_entry_start_pos, _ = struct.unpack("<IB", next_index_entry)
        entry_end_pos = next_entry_start_pos
    else:
        # Last entry - use data file size
        entry_end_pos = data_file_size

    # Calculate target token count
    # Entry format: [corpus_id(1B)] [line_number(4B)] [src_tokens(2B each)] [tgt_tokens(2B each)]
    tgt_start_pos = entry_start_pos + 1 + 4 + (src_token_count * 2)  # Skip header + src tokens
    tgt_byte_count = entry_end_pos - tgt_start_pos
    tgt_token_count = tgt_byte_count // 2  # Each token is 2 bytes

    return entry_start_pos, src_token_count, tgt_token_count


def read_from_data_file(
    data_file: BinaryIO,
    entry_start_pos: int,
    src_token_count: int,
    tgt_token_count: int,
) -> tuple[int, int, list[int], list[int]]:
    """
    Read a complete entry from the data file.

    Args:
        data_file: Open binary file handle for reading data
        entry_start_pos: Starting position of the entry in the data file
        src_token_count: Number of source tokens in the entry
        tgt_token_count: Number of target tokens in the entry

    Returns:
        Tuple of (corpus_id, original_line_number, src_tokens, tgt_tokens)
    """
    # Seek to the start of the entry
    data_file.seek(entry_start_pos)

    # Read corpus_id (1 byte unsigned int)
    corpus_id = struct.unpack("<B", data_file.read(1))[0]

    # Read original line number (4 bytes little endian unsigned int)
    original_line_number = struct.unpack("<I", data_file.read(4))[0]

    # Read source tokens (2 bytes each, little endian)
    src_tokens = []
    if src_token_count > 0:
        src_data = data_file.read(src_token_count * 2)
        src_tokens = list(struct.unpack("<" + "H" * src_token_count, src_data))

    # Read target tokens (2 bytes each, little endian)
    tgt_tokens = []
    if tgt_token_count > 0:
        tgt_data = data_file.read(tgt_token_count * 2)
        tgt_tokens = list(struct.unpack("<" + "H" * tgt_token_count, tgt_data))

    return corpus_id, original_line_number, src_tokens, tgt_tokens


def _process_index_chunk(
    data_file_path: str,
    index_file_path: str,
    chunk_start: int,
    chunk_size: int,
    entries_count: int,
    step_size: int,
    num_buckets: int,
) -> list[list[int]]:
    """
    Process a chunk of entries by reading only the assigned range.

    Args:
        data_file_path: Path to the data file
        index_file_path: Path to the index file
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

    with open(data_file_path, "rb") as data_file, open(index_file_path, "rb") as index_file:
        for entry_idx in range(chunk_start, chunk_end):
            # Read index entry
            index_file.seek(entry_idx * 5)
            index_entry = index_file.read(5)
            if not index_entry:
                break

            entry_start_pos, src_token_count = struct.unpack("<IB", index_entry)

            # Calculate target token count by finding the next entry or end of file
            if entry_idx + 1 < entries_count:
                # Read next entry's start position
                index_file.seek((entry_idx + 1) * 5)
                next_entry = index_file.read(5)
                next_entry_start_pos, _ = struct.unpack("<IB", next_entry)
                tgt_end_pos = next_entry_start_pos
            else:
                # Last entry - read to end of file
                data_file.seek(0, 2)  # Seek to end
                tgt_end_pos = data_file.tell()

            # Calculate target token count
            # Entry format: [corpus_id] [line_number] [src_tokens] [tgt_tokens]
            tgt_start_pos = entry_start_pos + 1 + 4 + (src_token_count * 2)
            tgt_token_count = (tgt_end_pos - tgt_start_pos) // 2

            effective_tgt_len = tgt_token_count + 1  # Add 1 for EOS token
            max_len = max(src_token_count, effective_tgt_len)

            bucket_idx = min((max_len + step_size - 1) // step_size - 1, num_buckets - 1)
            if bucket_idx < 0:
                bucket_idx = 0

            buckets[bucket_idx].append(entry_idx)

    return buckets


def create_chunked_index(
    dataset_file_path_prefix: str,
    step_size: int = 16,
    max_length: int = 512,
    num_processes: int | None = None,
) -> int:
    """
    Create a chunked index file that groups entry indices by maximum sentence length.

    The chunked index format:
    - Header: [num_buckets(4B)] [bucket_offset_1(4B)] [bucket_offset_2(4B)] ...
    - For each bucket: entry indices with format [entry_idx(4B)] = 4 bytes per entry

    Args:
        dataset_file_path: Path to the dataset file (without suffix)
        step_size: Step size for buckets (e.g., 16 means buckets: 16, 32, 48...)
        max_length: Maximum length to consider
        num_processes: Number of processes to use (None for CPU count)

    Returns:
        Total number of entries processed
    """
    data_file_path = dataset_file_path_prefix + ".bin"
    index_file_path = dataset_file_path_prefix + ".idx"
    output_index_path = dataset_file_path_prefix + ".cidx"  # chunked index

    if not os.path.exists(index_file_path):
        raise FileNotFoundError(f"Dataset data file not found: {index_file_path}")

    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Dataset data file not found: {data_file_path}")

    # Calculate number of buckets and entries
    num_buckets = (max_length + step_size - 1) // step_size
    index_file_size = os.path.getsize(index_file_path)
    entries_count = index_file_size // 5  # 5 bytes per entry in current format

    # Determine number of processes
    if num_processes is None:
        import multiprocessing

        num_processes = multiprocessing.cpu_count()

    # Parallel processing
    chunk_size = max(1, entries_count // num_processes)
    chunks = []

    for i in range(0, entries_count, chunk_size):
        remaining = entries_count - i
        actual_chunk_size = min(chunk_size, remaining)
        chunks.append((i, actual_chunk_size))

    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        worker_func = partial(
            _process_index_chunk,
            data_file_path,
            index_file_path,
            entries_count=entries_count,
            step_size=step_size,
            num_buckets=num_buckets,
        )

        # Submit all chunks
        results = list(pool.starmap(worker_func, chunks))

    # Merge results from all processes
    buckets: list[list[int]] = [[] for _ in range(num_buckets)]
    total_entries = 0
    for chunk_buckets in results:
        for bucket_idx, entries in enumerate(chunk_buckets):
            buckets[bucket_idx].extend(entries)
            total_entries += len(entries)

    # Write chunked index file
    with open(output_index_path, "wb") as output_file:
        # Write header: number of buckets followed by bucket offsets (to be filled later)
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


def read_chunked_index_header(chunked_index_path: str) -> tuple[int, list[int]]:
    """
    Read the header of a chunked index file to get bucket information.

    Args:
        chunked_index_path: Path to the chunked index file (.cidx)

    Returns:
        Tuple of (num_buckets, bucket_offsets)
    """
    with open(chunked_index_path, "rb") as f:
        num_buckets = struct.unpack("<I", f.read(4))[0]
        bucket_offsets = []
        for _ in range(num_buckets):
            offset = struct.unpack("<I", f.read(4))[0]
            bucket_offsets.append(offset)

    return num_buckets, bucket_offsets


def get_bucket_sizes(chunked_index_path: str) -> list[int]:
    """
    Get the number of entries in each bucket without reading the actual indices.

    Args:
        chunked_index_path: Path to the chunked index file (.cidx)

    Returns:
        List of entry counts for each bucket
    """
    num_buckets, bucket_offsets = read_chunked_index_header(chunked_index_path)

    bucket_sizes = []
    for i in range(num_buckets):
        start_offset = bucket_offsets[i]
        end_offset = bucket_offsets[i + 1] if i + 1 < num_buckets else None

        if end_offset is None:
            # Last bucket - need to check file size
            file_size = os.path.getsize(chunked_index_path)
            bucket_byte_size = file_size - start_offset
        else:
            bucket_byte_size = end_offset - start_offset

        # Each entry is 4 bytes, so divide by 4 to get entry count
        entry_count = bucket_byte_size // 4
        bucket_sizes.append(entry_count)

    return bucket_sizes


def print_bucket_summary(chunked_index_path: str, step_size: int = 16) -> None:
    """
    Print a summary of bucket sizes.

    Args:
        chunked_index_path: Path to the chunked index file (.cidx)
        step_size: Step size used when creating buckets
    """
    bucket_sizes = get_bucket_sizes(chunked_index_path)
    total_entries = sum(bucket_sizes)

    print(f"Bucket summary for {chunked_index_path}:")
    print(f"Total entries: {total_entries}")
    print()

    for i, count in enumerate(bucket_sizes):
        if count > 0:  # Only show non-empty buckets
            max_bucket_len = (i + 1) * step_size
            min_bucket_len = i * step_size + 1 if i > 0 else 1
            percentage = (count / total_entries * 100) if total_entries > 0 else 0
            print(
                f"  Bucket {i} ({min_bucket_len:3d}-{max_bucket_len:3d} tokens): "
                f"{count:6d} entries ({percentage:5.1f}%)"
            )


def get_entry_idx_from_bucket(
    chunked_index_file: BinaryIO,
    bucket_id: int,
    idx_in_bucket: int,
) -> int:
    """
    Get the entry index for a specific position within a bucket from a chunked index file.

    Args:
        chunked_index_file: Open binary file handle for reading chunked index
        bucket_id: ID of the bucket to read from
        idx_in_bucket: Index within the bucket (0-based)

    Returns:
        The entry index from the original dataset

    Raises:
        ValueError: If bucket_id is invalid or idx_in_bucket is out of range
    """
    # Read header to get bucket information
    chunked_index_file.seek(0)
    num_buckets = struct.unpack("<I", chunked_index_file.read(4))[0]

    if bucket_id >= num_buckets:
        raise ValueError(f"Bucket ID {bucket_id} >= number of buckets {num_buckets}")

    # Read the specific bucket offset we need
    chunked_index_file.seek(4 + bucket_id * 4)  # Skip num_buckets + previous bucket offsets
    bucket_start_offset = struct.unpack("<I", chunked_index_file.read(4))[0]

    # Jump directly to the entry within the bucket
    entry_offset = bucket_start_offset + (idx_in_bucket * 4)
    chunked_index_file.seek(entry_offset)

    # Read and return the entry index
    entry_data = chunked_index_file.read(4)
    if len(entry_data) < 4:
        raise ValueError(f"Could not read entry at bucket {bucket_id}, index {idx_in_bucket}")

    entry_idx: int = struct.unpack("<I", entry_data)[0]
    return entry_idx


def get_number_of_entries(dataset_file_path: str) -> int:
    """
    Get the number of entries in a binary dataset.

    Args:
        dataset_file_path: Path to the dataset file (without suffix)

    Returns:
        Number of entries in the dataset
    """
    index_file_path = dataset_file_path + ".idx"
    index_file_size = os.path.getsize(index_file_path)
    return index_file_size // 5  # Each index entry is 5 bytes


def split_dataset(
    input_file_path: str,
    output_file_path_a: str,
    output_file_path_b: str,
    num_samples_in_a: int,
) -> None:
    """
    Split a dataset into two parts.

    Args:
        input_file_path: Path to the input dataset file (without suffix)
        output_file_a_path: Path to the first output dataset file (without suffix)
        output_file_b_path: Path to the second output dataset file (without suffix)
        num_samples_in_a: Number of samples to include in the first output file
    """
    input_index_file_path = input_file_path + ".idx"
    input_data_file_path = input_file_path + ".bin"

    num_entries = get_number_of_entries(input_file_path)

    # Randomly select indices to split the dataset
    a_indices = random.sample(range(num_entries), num_samples_in_a)

    # Get size of data file
    data_file_size = os.path.getsize(input_data_file_path)

    with open(input_index_file_path, "rb") as input_index, open(
        input_data_file_path, "rb"
    ) as input_data, open(output_file_path_a + ".idx", "wb") as a_output_index, open(
        output_file_path_b + ".idx", "wb"
    ) as b_output_index, open(
        output_file_path_a + ".bin", "wb"
    ) as a_output_data, open(
        output_file_path_b + ".bin", "wb"
    ) as b_output_data:

        # Convert a_indices to a set for faster lookup
        a_indices_set = set(a_indices)

        for i in trange(num_entries):
            # Determine which output files to use
            if i in a_indices_set:
                output_data = a_output_data
                output_index = a_output_index
            else:
                output_data = b_output_data
                output_index = b_output_index

            # Get entry information from index
            entry_start_pos, src_token_count, tgt_token_count = get_entry_info_from_index(
                input_index, data_file_size, i
            )

            # Calculate the number of bytes to read for this entry
            num_bytes_to_read = 1 + 4 + src_token_count * 2 + tgt_token_count * 2

            # Read the complete data entry from input
            input_data.seek(entry_start_pos)
            data_entry = input_data.read(num_bytes_to_read)
            if len(data_entry) < num_bytes_to_read:
                raise ValueError(
                    f"Data entry {i} is incomplete: expected {num_bytes_to_read} bytes, "
                    f"got {len(data_entry)}"
                )

            # Get current position in output data file for the new index entry
            new_entry_start_pos = output_data.tell()

            # Write the data entry to the appropriate output file
            output_data.write(data_entry)

            # Write the updated index entry (new position + source token count)
            output_index.write(struct.pack("<IB", new_entry_start_pos, src_token_count))
