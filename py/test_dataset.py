import os
import struct
import tempfile

import pytest

from dataset import (
    Dataset,
    append_to_dataset,
    concatenate_datasets,
    create_subset,
    get_entry,
    open_dataset,
    split_dataset,
)
from dataset_test_helpers import make_toy_dataset, make_toy_tokens


def test_append_and_get_entry() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "toy")
        src_idx = open(base + ".src.idx", "wb+")
        src_bin = open(base + ".src.bin", "wb+")
        tgt_idx = open(base + ".tgt.idx", "wb+")
        tgt_bin = open(base + ".tgt.bin", "wb+")
        meta = open(base + ".meta", "wb+")
        dataset = Dataset(src_idx, src_bin, tgt_idx, tgt_bin, meta, 0, 0, 0)
        # Add 3 entries
        for i in range(3):
            append_to_dataset(
                dataset,
                corpus_id=i,
                original_line_number=100 + i,
                src_tokens=make_toy_tokens(i, 4),
                tgt_tokens=make_toy_tokens(i + 10, 5),
            )
        # Flush and close for reading
        src_idx.close()
        src_bin.close()
        tgt_idx.close()
        tgt_bin.close()
        meta.close()
        with open_dataset(base) as ds:
            for i in range(3):
                corpus_id, orig_line, src, tgt = get_entry(ds, i)
                assert corpus_id == i
                assert orig_line == 100 + i
                assert src == make_toy_tokens(i, 4)
                assert tgt == make_toy_tokens(i + 10, 5)


def test_concatenate_datasets() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bases = []
        for d in range(2):
            base = os.path.join(tmpdir, f"toy{d}")
            src_idx = open(base + ".src.idx", "wb+")
            src_bin = open(base + ".src.bin", "wb+")
            tgt_idx = open(base + ".tgt.idx", "wb+")
            tgt_bin = open(base + ".tgt.bin", "wb+")
            meta = open(base + ".meta", "wb+")
            dataset = Dataset(src_idx, src_bin, tgt_idx, tgt_bin, meta, 0, 0, 0)
            for i in range(2):
                append_to_dataset(
                    dataset,
                    corpus_id=d,
                    original_line_number=10 * d + i,
                    src_tokens=make_toy_tokens(100 * d + i, 3),
                    tgt_tokens=make_toy_tokens(200 * d + i, 2),
                )
            src_idx.close()
            src_bin.close()
            tgt_idx.close()
            tgt_bin.close()
            meta.close()
            bases.append(base)
        out_base = os.path.join(tmpdir, "combined")
        concatenate_datasets(out_base, bases)
        with open_dataset(out_base) as ds:
            assert ds.num_entries == 4
            # Check all entries
            expected = []
            for d in range(2):
                for i in range(2):
                    expected.append(
                        (
                            d,
                            10 * d + i,
                            make_toy_tokens(100 * d + i, 3),
                            make_toy_tokens(200 * d + i, 2),
                        )
                    )
            for i in range(4):
                assert get_entry(ds, i) == expected[i]


def test_append_to_dataset_invalid_tokens() -> None:
    """Test that append_to_dataset raises error for negative or >16bit token values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "bad")
        src_idx = open(base + ".src.idx", "wb+")
        src_bin = open(base + ".src.bin", "wb+")
        tgt_idx = open(base + ".tgt.idx", "wb+")
        tgt_bin = open(base + ".tgt.bin", "wb+")
        meta = open(base + ".meta", "wb+")
        dataset = Dataset(src_idx, src_bin, tgt_idx, tgt_bin, meta, 0, 0, 0)
        # Negative token
        with pytest.raises(struct.error):
            append_to_dataset(dataset, 0, 0, [-1, 2, 3], [1, 2])
        # >16bit token
        with pytest.raises(struct.error):
            append_to_dataset(dataset, 0, 0, [1, 2], [70000, 2])
        src_idx.close()
        src_bin.close()
        tgt_idx.close()
        tgt_bin.close()
        meta.close()


def test_split_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "toy")
        entries = [([1, 2], [3, 4]), ([5, 6, 7], [8]), ([9], [10, 11, 12, 13]), ([14], [15])]
        make_toy_dataset(prefix, entries)
        out_a = os.path.join(tmpdir, "a")
        out_b = os.path.join(tmpdir, "b")
        split_dataset(prefix, out_a, out_b, num_samples_in_a=2)
        # Check that both splits have the right number of entries
        for out, expected_count in [(out_a, 2), (out_b, 2)]:
            with open(out + ".src.idx", "rb") as src_idx:
                src_idx.seek(0, 2)
                count = src_idx.tell() // 4
                assert count == expected_count
            with open(out + ".tgt.idx", "rb") as tgt_idx:
                tgt_idx.seek(0, 2)
                count = tgt_idx.tell() // 4
                assert count == expected_count
            with open(out + ".meta", "rb") as meta:
                meta.seek(0, 2)
                count = meta.tell() // 5
                assert count == expected_count
        # Optionally, check that all original entries are present in one of the splits
        found = set()
        for out in [out_a, out_b]:
            with open(out + ".src.idx", "rb") as src_idx:
                src_idx.seek(0, 2)
                count = src_idx.tell() // 4
            with open(out + ".meta", "rb") as meta:
                for i in range(count):
                    meta.seek(i * 5)
                    meta_bytes = meta.read(5)
                    _, orig_line = int.from_bytes(meta_bytes[:1], "little"), int.from_bytes(
                        meta_bytes[1:], "little"
                    )
                    found.add(orig_line)
        assert found == set(range(4))


def test_create_subset() -> None:
    """Test create_subset function for efficient random subset sampling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "toy")
        entries = [([1, 2], [3, 4]), ([5, 6, 7], [8]), ([9], [10, 11, 12, 13]), ([14], [15])]
        make_toy_dataset(prefix, entries)

        out_subset = os.path.join(tmpdir, "subset")
        # Create subset with 2 samples
        create_subset(prefix, out_subset, num_samples=2)

        # Check that all files were created
        assert os.path.exists(out_subset + ".src.idx")
        assert os.path.exists(out_subset + ".src.bin")
        assert os.path.exists(out_subset + ".tgt.idx")
        assert os.path.exists(out_subset + ".tgt.bin")
        assert os.path.exists(out_subset + ".meta")

        # Check that the subset has the correct number of entries
        with open(out_subset + ".src.idx", "rb") as src_idx:
            src_idx.seek(0, 2)
            count = src_idx.tell() // 4
            assert count == 2

        with open(out_subset + ".tgt.idx", "rb") as tgt_idx:
            tgt_idx.seek(0, 2)
            count = tgt_idx.tell() // 4
            assert count == 2

        with open(out_subset + ".meta", "rb") as meta:
            meta.seek(0, 2)
            count = meta.tell() // 5
            assert count == 2

        # Verify the sampled entries are valid (can be read correctly)
        with open_dataset(out_subset) as ds:
            assert ds.num_entries == 2
            for i in range(2):
                corpus_id, orig_line, src, tgt = get_entry(ds, i)
                assert isinstance(corpus_id, int)
                assert isinstance(orig_line, int)
                assert isinstance(src, list)
                assert isinstance(tgt, list)
                # Original line should be one of 0, 1, 2, 3 from the original dataset
                assert orig_line in range(4)


def test_create_subset_error_cases() -> None:
    """Test create_subset error handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "toy")
        entries = [([1, 2], [3]), ([4, 5], [6])]
        make_toy_dataset(prefix, entries)

        out_subset = os.path.join(tmpdir, "subset")

        # Test error when requesting more samples than available
        with pytest.raises(
            ValueError, match="Cannot sample 5 entries from dataset with only 2 entries"
        ):
            create_subset(prefix, out_subset, num_samples=5)

        # Test successful sampling of all entries
        create_subset(prefix, out_subset, num_samples=2)
        with open_dataset(out_subset) as ds:
            assert ds.num_entries == 2
