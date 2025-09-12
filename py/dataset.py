import random
import shutil
import struct
from contextlib import contextmanager
from typing import BinaryIO, Generator

from tqdm import trange

from data import from_uint16_le_bytes, to_uint16_le_bytes
from fileio import read_uint32, write_uint32
from indexed_sequential import (
    append_indexed_entry,
    concatenate_index_files,
)


class Dataset:
    src_idx_file: BinaryIO
    src_data_file: BinaryIO
    tgt_idx_file: BinaryIO
    tgt_data_file: BinaryIO
    meta_file: BinaryIO
    src_data_size: int
    tgt_data_size: int
    num_entries: int

    def __init__(
        self,
        src_idx_file: BinaryIO,
        src_data_file: BinaryIO,
        tgt_idx_file: BinaryIO,
        tgt_data_file: BinaryIO,
        meta_file: BinaryIO,
        src_data_size: int,
        tgt_data_size: int,
        num_entries: int,
    ):
        self.src_idx_file = src_idx_file
        self.src_data_file = src_data_file
        self.tgt_idx_file = tgt_idx_file
        self.tgt_data_file = tgt_data_file
        self.meta_file = meta_file
        self.src_data_size = src_data_size
        self.tgt_data_size = tgt_data_size
        self.num_entries = num_entries


@contextmanager
def open_dataset(base_path: str) -> Generator[Dataset, None, None]:
    src_idx_file = open(f"{base_path}.src.idx", "rb")
    src_data_file = open(f"{base_path}.src.bin", "rb")
    tgt_idx_file = open(f"{base_path}.tgt.idx", "rb")
    tgt_data_file = open(f"{base_path}.tgt.bin", "rb")
    meta_file = open(f"{base_path}.meta", "rb")
    try:
        src_data_file.seek(0, 2)
        src_data_size = src_data_file.tell()
        tgt_data_file.seek(0, 2)
        tgt_data_size = tgt_data_file.tell()
        src_idx_file.seek(0, 2)
        num_entries = src_idx_file.tell() // 4
        src_idx_file.seek(0)
        yield Dataset(
            src_idx_file=src_idx_file,
            src_data_file=src_data_file,
            tgt_idx_file=tgt_idx_file,
            tgt_data_file=tgt_data_file,
            meta_file=meta_file,
            src_data_size=src_data_size,
            tgt_data_size=tgt_data_size,
            num_entries=num_entries,
        )
    finally:
        src_idx_file.close()
        src_data_file.close()
        tgt_idx_file.close()
        tgt_data_file.close()
        meta_file.close()


def append_to_dataset(
    dataset: "Dataset",
    corpus_id: int,
    original_line_number: int,
    src_tokens: list[int],
    tgt_tokens: list[int],
) -> None:
    """
    Append a single entry to the new dataset format using a Dataset object.
    Args:
        dataset: Dataset object with open files
        corpus_id: ID of the corpus (1 byte unsigned int)
        original_line_number: Original line number in source file
        src_tokens: List of source token IDs
        tgt_tokens: List of target token IDs
    """
    src_bytes = to_uint16_le_bytes(src_tokens)
    tgt_bytes = to_uint16_le_bytes(tgt_tokens)
    append_indexed_entry(dataset.src_data_file, dataset.src_idx_file, src_bytes)
    append_indexed_entry(dataset.tgt_data_file, dataset.tgt_idx_file, tgt_bytes)
    dataset.meta_file.write(struct.pack("<BI", corpus_id, original_line_number))


def get_entry(dataset: "Dataset", entry_idx: int) -> tuple[int, int, list[int], list[int]]:
    """
    Get a complete entry from the new dataset format using a Dataset object.
    Args:
        dataset: Dataset object with open files
        entry_idx: Index of the entry to read
    Returns:
        Tuple of (corpus_id, original_line_number, src_tokens, tgt_tokens)
    """
    # Read src tokens
    dataset.src_idx_file.seek(entry_idx * 4)
    src_end = read_uint32(dataset.src_idx_file)
    src_start = 0
    if entry_idx > 0:
        dataset.src_idx_file.seek((entry_idx - 1) * 4)
        src_start = read_uint32(dataset.src_idx_file)
    dataset.src_data_file.seek(src_start)
    src_token_bytes = dataset.src_data_file.read(src_end - src_start)
    src_tokens = from_uint16_le_bytes(src_token_bytes)

    # Read tgt tokens
    dataset.tgt_idx_file.seek(entry_idx * 4)
    tgt_end = read_uint32(dataset.tgt_idx_file)
    tgt_start = 0
    if entry_idx > 0:
        dataset.tgt_idx_file.seek((entry_idx - 1) * 4)
        tgt_start = read_uint32(dataset.tgt_idx_file)
    dataset.tgt_data_file.seek(tgt_start)
    tgt_token_bytes = dataset.tgt_data_file.read(tgt_end - tgt_start)
    tgt_tokens = from_uint16_le_bytes(tgt_token_bytes)

    # Read metadata
    dataset.meta_file.seek(entry_idx * 5)
    meta_bytes = dataset.meta_file.read(5)
    corpus_id, original_line_number = struct.unpack("<BI", meta_bytes)

    return corpus_id, original_line_number, src_tokens, tgt_tokens


def get_entries(
    dataset: "Dataset", entry_indices: list[int]
) -> list[tuple[int, int, list[int], list[int]]]:
    """
    Get multiple entries from the dataset using a Dataset object and a list of indices.
    Args:
        dataset: Dataset object with open files
        entry_indices: List of entry indices to read
    Returns:
        List of tuples (corpus_id, original_line_number, src_tokens, tgt_tokens)
    """
    return [get_entry(dataset, idx) for idx in entry_indices]


def concatenate_datasets(output_prefix: str, input_prefixes: list[str]) -> None:
    """
    Concatenate multiple datasets (src/tgt indexed files and metadata file).

    Args:
        output_prefix: Path prefix for the combined dataset (no extension)
        input_prefixes: List of path prefixes for input datasets (no extension)
    """
    # Concatenate .src.bin and .tgt.bin
    for ext in ["src.bin", "tgt.bin"]:
        out_path = f"{output_prefix}.{ext}"
        with open(out_path, "wb") as out_f:
            for prefix in input_prefixes:
                in_path = f"{prefix}.{ext}"
                with open(in_path, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, length=1024 * 1024)

    # Concatenate .meta
    out_meta = f"{output_prefix}.meta"
    with open(out_meta, "wb") as out_f:
        for prefix in input_prefixes:
            in_path = f"{prefix}.meta"
            with open(in_path, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=1024 * 1024)

    # Concatenate .src.idx and .tgt.idx with offset adjustment
    for ext in ["src.idx", "tgt.idx"]:
        index_files = [f"{prefix}.{ext}" for prefix in input_prefixes]
        out_index = f"{output_prefix}.{ext}"
        concatenate_index_files(index_files, out_index)


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
        output_file_path_a: Path to the first output dataset file (without suffix)
        output_file_path_b: Path to the second output dataset file (without suffix)
        num_samples_in_a: Number of samples to include in the first output file
    """
    # New format: split src.idx/src.bin, tgt.idx/tgt.bin, meta
    src_idx_path = input_file_path + ".src.idx"
    src_bin_path = input_file_path + ".src.bin"
    tgt_idx_path = input_file_path + ".tgt.idx"
    tgt_bin_path = input_file_path + ".tgt.bin"
    meta_path = input_file_path + ".meta"

    # Count entries
    with open(src_idx_path, "rb") as src_idx:
        src_idx.seek(0, 2)
        num_entries = src_idx.tell() // 4

    # Randomly select indices to split the dataset
    a_indices = set(random.sample(range(num_entries), num_samples_in_a))

    # Open all files
    with open(src_idx_path, "rb") as src_idx, open(src_bin_path, "rb") as src_bin, open(
        tgt_idx_path, "rb"
    ) as tgt_idx, open(tgt_bin_path, "rb") as tgt_bin, open(meta_path, "rb") as meta, open(
        output_file_path_a + ".src.idx", "wb"
    ) as a_src_idx, open(
        output_file_path_a + ".src.bin", "wb"
    ) as a_src_bin, open(
        output_file_path_a + ".tgt.idx", "wb"
    ) as a_tgt_idx, open(
        output_file_path_a + ".tgt.bin", "wb"
    ) as a_tgt_bin, open(
        output_file_path_a + ".meta", "wb"
    ) as a_meta, open(
        output_file_path_b + ".src.idx", "wb"
    ) as b_src_idx, open(
        output_file_path_b + ".src.bin", "wb"
    ) as b_src_bin, open(
        output_file_path_b + ".tgt.idx", "wb"
    ) as b_tgt_idx, open(
        output_file_path_b + ".tgt.bin", "wb"
    ) as b_tgt_bin, open(
        output_file_path_b + ".meta", "wb"
    ) as b_meta:

        a_src_offset = 0
        a_tgt_offset = 0
        b_src_offset = 0
        b_tgt_offset = 0

        for i in trange(num_entries):
            # Read src tokens
            src_idx.seek(i * 4)
            src_end = read_uint32(src_idx)
            src_start = 0
            if i > 0:
                src_idx.seek((i - 1) * 4)
                src_start = read_uint32(src_idx)
            src_bin.seek(src_start)
            src_bytes = src_bin.read(src_end - src_start)

            # Read tgt tokens
            tgt_idx.seek(i * 4)
            tgt_end = read_uint32(tgt_idx)
            tgt_start = 0
            if i > 0:
                tgt_idx.seek((i - 1) * 4)
                tgt_start = read_uint32(tgt_idx)
            tgt_bin.seek(tgt_start)
            tgt_bytes = tgt_bin.read(tgt_end - tgt_start)

            # Read meta
            meta.seek(i * 5)
            meta_bytes = meta.read(5)

            if i in a_indices:
                # Write src
                a_src_bin.write(src_bytes)
                a_src_offset += len(src_bytes)
                write_uint32(a_src_idx, a_src_offset)
                # Write tgt
                a_tgt_bin.write(tgt_bytes)
                a_tgt_offset += len(tgt_bytes)
                write_uint32(a_tgt_idx, a_tgt_offset)
                # Write meta
                a_meta.write(meta_bytes)
            else:
                b_src_bin.write(src_bytes)
                b_src_offset += len(src_bytes)
                write_uint32(b_src_idx, b_src_offset)
                b_tgt_bin.write(tgt_bytes)
                b_tgt_offset += len(tgt_bytes)
                write_uint32(b_tgt_idx, b_tgt_offset)
                b_meta.write(meta_bytes)


def create_subset(
    input_file_path: str,
    output_file_path: str,
    num_samples: int,
) -> None:
    """
    Create a random subset of a dataset by only reading the required entries (raw binary copy).

    Args:
        input_file_path: Path to the input dataset file (without suffix)
        output_file_path: Path to the output dataset file (without suffix)
        num_samples: Number of samples to include in the subset
    """
    # File paths
    src_idx_path = input_file_path + ".src.idx"
    src_bin_path = input_file_path + ".src.bin"
    tgt_idx_path = input_file_path + ".tgt.idx"
    tgt_bin_path = input_file_path + ".tgt.bin"
    meta_path = input_file_path + ".meta"

    # Get total number of entries
    with open(src_idx_path, "rb") as src_idx:
        src_idx.seek(0, 2)
        num_entries = src_idx.tell() // 4

    if num_samples > num_entries:
        raise ValueError(
            f"Cannot sample {num_samples} entries from dataset with only {num_entries} entries"
        )

    # Randomly select indices to include in subset
    selected_indices = sorted(random.sample(range(num_entries), num_samples))

    # Open all input and output files
    with open(src_idx_path, "rb") as src_idx, open(src_bin_path, "rb") as src_bin, open(
        tgt_idx_path, "rb"
    ) as tgt_idx, open(tgt_bin_path, "rb") as tgt_bin, open(meta_path, "rb") as meta, open(
        output_file_path + ".src.idx", "wb"
    ) as out_src_idx, open(
        output_file_path + ".src.bin", "wb"
    ) as out_src_bin, open(
        output_file_path + ".tgt.idx", "wb"
    ) as out_tgt_idx, open(
        output_file_path + ".tgt.bin", "wb"
    ) as out_tgt_bin, open(
        output_file_path + ".meta", "wb"
    ) as out_meta:

        src_offset = 0
        tgt_offset = 0

        # Process only the selected entries
        for idx in trange(len(selected_indices), desc="Creating subset"):
            entry_idx = selected_indices[idx]

            # Read src data directly (raw bytes)
            src_idx.seek(entry_idx * 4)
            src_end = read_uint32(src_idx)
            src_start = 0
            if entry_idx > 0:
                src_idx.seek((entry_idx - 1) * 4)
                src_start = read_uint32(src_idx)
            src_bin.seek(src_start)
            src_bytes = src_bin.read(src_end - src_start)

            # Read tgt data directly (raw bytes)
            tgt_idx.seek(entry_idx * 4)
            tgt_end = read_uint32(tgt_idx)
            tgt_start = 0
            if entry_idx > 0:
                tgt_idx.seek((entry_idx - 1) * 4)
                tgt_start = read_uint32(tgt_idx)
            tgt_bin.seek(tgt_start)
            tgt_bytes = tgt_bin.read(tgt_end - tgt_start)

            # Read meta data directly (raw bytes)
            meta.seek(entry_idx * 5)
            meta_bytes = meta.read(5)

            # Write to output files (raw binary copy)
            out_src_bin.write(src_bytes)
            src_offset += len(src_bytes)
            write_uint32(out_src_idx, src_offset)

            out_tgt_bin.write(tgt_bytes)
            tgt_offset += len(tgt_bytes)
            write_uint32(out_tgt_idx, tgt_offset)

            out_meta.write(meta_bytes)
