import struct
from typing import BinaryIO

from fileio import read_uint32, write_uint32


def concatenate_index_files(index_files: list[str], output_index_file: str) -> None:
    """
    Concatenate index files, adjusting offsets by cumulative end position.
    Args:
        index_files: list of index file paths (in order)
        output_index_file: output index file path
    """
    offset = 0
    with open(output_index_file, "wb") as out_f:
        for idx_file in index_files:
            with open(idx_file, "rb") as in_f:
                entries = []
                while True:
                    entry = in_f.read(4)
                    if not entry:
                        break
                    end_pos = struct.unpack("<I", entry)[0]
                    entries.append(end_pos)
                for end_pos in entries:
                    write_uint32(out_f, end_pos + offset)
                if entries:
                    offset += entries[-1]


def append_indexed_entry(data_file: BinaryIO, index_file: BinaryIO, entry: bytes) -> None:
    """
    Append a single entry to already opened data and index files.
    """
    data_file.seek(0, 2)  # Seek to end
    data_file.write(entry)
    offset = data_file.tell()
    write_uint32(index_file, offset)


def read_indexed_entry(data_file: BinaryIO, index_file: BinaryIO, entry_idx: int) -> bytes:
    """
    Read a single entry from already opened data and index files.
    """
    index_file.seek(entry_idx * 4)
    end_offset = read_uint32(index_file)
    start_offset = 0
    if entry_idx > 0:
        index_file.seek((entry_idx - 1) * 4)
        start_offset = read_uint32(index_file)
    data_file.seek(start_offset)
    return data_file.read(end_offset - start_offset)
