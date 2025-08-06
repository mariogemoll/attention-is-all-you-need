import os
import struct
from typing import BinaryIO

from tqdm import tqdm


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
