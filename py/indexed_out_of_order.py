from typing import BinaryIO

from fileio import read_uint32, write_uint32


def add_entry(index_file: BinaryIO, data_file: BinaryIO, idx: int, data: bytes) -> None:
    """
    Append data to the end of the data file and record its start and end
    offsets in the index file at position `idx`.

    Format: each index entry is 8 bytes (little-endian):
        - 4 bytes: start offset (uint32)
        - 4 bytes: end offset   (uint32)

    The function supports out-of-order writes by seeking to `idx * 8` in the
    index file. If this seeks beyond EOF, the index file is extended.
    """
    # Write the data at the end of the data file
    data_file.seek(0, 2)
    start = data_file.tell()
    data_file.write(data)
    end = data_file.tell()

    # Write start and end offsets into the index file at the specified slot
    index_file.seek(idx * 8)
    write_uint32(index_file, start)
    write_uint32(index_file, end)


def read_entry(index_file: BinaryIO, data_file: BinaryIO, idx: int) -> bytes:
    """
    Read a single entry from a start/end-indexed pair of files.

    Expects each index entry to be 8 bytes (start, end as uint32 LE).
    """
    index_file.seek(idx * 8)
    start = read_uint32(index_file)
    end = read_uint32(index_file)
    if end < start:
        raise ValueError("Invalid index entry: end < start")
    data_file.seek(start)
    return data_file.read(end - start)


def convert_to_sequential(
    in_idx: BinaryIO, in_data: BinaryIO, out_idx: BinaryIO, out_data: BinaryIO
) -> None:
    """
    Convert an out-of-order indexed pair (start/end per entry) into the
    sequential index format (end-only per entry) while rewriting the data file
    in logical index order.

    Inputs:
      - in_index_path: path to start/end index file (8 bytes per entry)
      - in_data_path: path to original data file referenced by the index
    Outputs:
      - out_index_path: path to write sequential index (4 bytes per entry)
      - out_data_path: path to write data in index order
    """
    # Determine number of entries
    in_idx.seek(0, 2)
    num_entries = in_idx.tell() // 8
    in_idx.seek(0)

    # Copy entries in index order into new data file and emit cumulative ends
    cumulative_end = 0
    for i in range(num_entries):
        in_idx.seek(i * 8)
        start = read_uint32(in_idx)
        end = read_uint32(in_idx)
        if end < start:
            raise ValueError(f"Invalid entry {i}: end < start")
        length = end - start
        if length > 0:
            in_data.seek(start)
            chunk = in_data.read(length)
            if len(chunk) != length:
                raise IOError(f"Failed to read {length} bytes for entry {i}")
            out_data.write(chunk)
        cumulative_end += length
        write_uint32(out_idx, cumulative_end)
