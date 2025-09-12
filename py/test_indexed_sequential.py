import os
import struct
import tempfile

from indexed_sequential import (
    append_indexed_entry,
    concatenate_index_files,
    read_indexed_entry,
)


def test_append_and_read_indexed_entry() -> None:
    entries = [b"abc", b"defg", b"hi"]
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "test.bin")
        idx_path = os.path.join(tmpdir, "test.idx")
        # Write entries one by one
        with open(data_path, "wb") as data_f, open(idx_path, "wb") as idx_f:
            for entry in entries:
                append_indexed_entry(data_f, idx_f, entry)
        # Read entries one by one
        with open(data_path, "rb") as data_f, open(idx_path, "rb") as idx_f:
            for i, expected in enumerate(entries):
                got = read_indexed_entry(data_f, idx_f, i)
                assert got == expected
        # Check that the last 4 bytes in the index file is the length of the data file
        data_size = os.path.getsize(data_path)
        with open(idx_path, "rb") as idx_f:
            idx_f.seek(-4, 2)
            last_offset = struct.unpack("<I", idx_f.read(4))[0]
            assert last_offset == data_size


def test_concatenate_index_files() -> None:
    # Create 3 small indexed files with known entries
    entries_list = [[b"a", b"bb"], [b"ccc"], [b"d", b"ee", b"fff"]]
    with tempfile.TemporaryDirectory() as tmpdir:
        index_paths = []
        data_paths = []
        all_entries = []
        for i, entries in enumerate(entries_list):
            data_path = os.path.join(tmpdir, f"data_{i}.bin")
            idx_path = os.path.join(tmpdir, f"data_{i}.idx")
            data_paths.append(data_path)
            index_paths.append(idx_path)
            with open(data_path, "wb") as data_f, open(idx_path, "wb") as idx_f:
                for entry in entries:
                    append_indexed_entry(data_f, idx_f, entry)
            all_entries.extend(entries)
        # Concatenate index files
        concat_idx = os.path.join(tmpdir, "concat.idx")
        concatenate_index_files(index_paths, concat_idx)
        # Concatenate data files
        concat_data = os.path.join(tmpdir, "concat.bin")
        with open(concat_data, "wb") as out_f:
            for data_path in data_paths:
                with open(data_path, "rb") as in_f:
                    out_f.write(in_f.read())
        # Check that all entries can be read back in order
        with open(concat_data, "rb") as data_f, open(concat_idx, "rb") as idx_f:
            for i, expected in enumerate(all_entries):
                got = read_indexed_entry(data_f, idx_f, i)
                assert got == expected
        # Check last offset matches data file size
        data_size = os.path.getsize(concat_data)
        with open(concat_idx, "rb") as idx_f:
            idx_f.seek(-4, 2)
            last_offset = struct.unpack("<I", idx_f.read(4))[0]
            assert last_offset == data_size
