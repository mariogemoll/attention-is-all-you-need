import os
import tempfile

from indexed_out_of_order import add_entry, convert_to_sequential, read_entry
from indexed_sequential import read_indexed_entry


def test_add_and_read_in_order() -> None:
    entries = [b"abc", b"defg", b"hi"]
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "test_se.bin")
        idx_path = os.path.join(tmpdir, "test_se.idx")
        with open(data_path, "wb+") as data_f, open(idx_path, "wb+") as idx_f:
            for i, entry in enumerate(entries):
                add_entry(idx_f, data_f, i, entry)

        with open(data_path, "rb") as data_f, open(idx_path, "rb") as idx_f:
            for i, expected in enumerate(entries):
                got = read_entry(idx_f, data_f, i)
                assert got == expected


def test_add_and_read_out_of_order() -> None:
    # Write entries out of order and ensure reading by idx is correct
    entries = [b"zero", b"one", b"two", b"three"]
    order = [1, 0, 3, 2]
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "test_se2.bin")
        idx_path = os.path.join(tmpdir, "test_se2.idx")
        with open(data_path, "wb+") as data_f, open(idx_path, "wb+") as idx_f:
            for idx in order:
                add_entry(idx_f, data_f, idx, entries[idx])

        with open(data_path, "rb") as data_f, open(idx_path, "rb") as idx_f:
            for i, expected in enumerate(entries):
                got = read_entry(idx_f, data_f, i)
                assert got == expected


def test_convert_to_sequential_roundtrip() -> None:
    entries = [b"zero", b"one", b"two", b"three"]
    order = [2, 0, 3, 1]  # write out of order
    with tempfile.TemporaryDirectory() as tmpdir:
        in_data = os.path.join(tmpdir, "in.bin")
        in_idx = os.path.join(tmpdir, "in.idx")
        with open(in_data, "wb+") as data_f, open(in_idx, "wb+") as idx_f:
            for idx in order:
                add_entry(idx_f, data_f, idx, entries[idx])

        out_data = os.path.join(tmpdir, "out.bin")
        out_idx = os.path.join(tmpdir, "out.idx")
        with open(in_idx, "rb") as in_idx_f, open(in_data, "rb") as in_data_f, open(
            out_idx, "wb"
        ) as out_idx_f, open(out_data, "wb") as out_data_f:
            convert_to_sequential(in_idx_f, in_data_f, out_idx_f, out_data_f)

        with open(out_data, "rb") as data_f, open(out_idx, "rb") as idx_f:
            for i, expected in enumerate(entries):
                got = read_indexed_entry(data_f, idx_f, i)
                assert got == expected
