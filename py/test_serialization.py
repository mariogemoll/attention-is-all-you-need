import io
import os
import struct
import tempfile

from serialization import (
    append_to_dataset,
    combine_datasets,
    create_chunked_index,
    get_entry_idx_from_bucket,
    get_entry_info_from_index,
)


def test_append_to_dataset_basic() -> None:
    """Test basic functionality of append_to_dataset."""
    # Create in-memory binary files
    data_file = io.BytesIO()
    index_file = io.BytesIO()

    # Test data
    corpus_id = 1
    original_line_number = 42
    src_tokens = [100, 200, 300]
    tgt_tokens = [400, 500]

    # Call the function
    append_to_dataset(
        data_file, index_file, corpus_id, original_line_number, src_tokens, tgt_tokens
    )

    # Verify data file contents
    data_file.seek(0)

    # Expected structure:
    # - corpus_id (1 byte): 1
    # - original_line_number (4 bytes): 42
    # - src_tokens (6 bytes): 100, 200, 300 (each 2 bytes)
    # - tgt_tokens (4 bytes): 400, 500 (each 2 bytes)

    # Parse and verify
    data_file.seek(0)

    # Read corpus_id
    parsed_corpus_id = struct.unpack("<B", data_file.read(1))[0]
    assert parsed_corpus_id == corpus_id

    # Read original_line_number
    parsed_line_number = struct.unpack("<I", data_file.read(4))[0]
    assert parsed_line_number == original_line_number

    # Read src_tokens
    src_bytes = data_file.read(2 * len(src_tokens))
    parsed_src_tokens = list(struct.unpack("<" + "H" * len(src_tokens), src_bytes))
    assert parsed_src_tokens == src_tokens

    # Read tgt_tokens
    tgt_bytes = data_file.read(2 * len(tgt_tokens))
    parsed_tgt_tokens = list(struct.unpack("<" + "H" * len(tgt_tokens), tgt_bytes))
    assert parsed_tgt_tokens == tgt_tokens

    # Verify index file contents
    index_file.seek(0)
    index_content = index_file.getvalue()

    # Should contain entry position (4 bytes) and source token count (1 byte)
    assert len(index_content) == 5

    entry_start_pos, src_token_count = struct.unpack("<IB", index_content)
    assert entry_start_pos == 0  # First entry starts at position 0
    assert src_token_count == len(src_tokens)  # Should match source token count


def test_append_to_dataset_empty_tokens() -> None:
    """Test with empty token lists."""
    data_file = io.BytesIO()
    index_file = io.BytesIO()

    corpus_id = 2
    original_line_number = 123
    src_tokens: list[int] = []
    tgt_tokens: list[int] = []

    append_to_dataset(
        data_file, index_file, corpus_id, original_line_number, src_tokens, tgt_tokens
    )

    # Verify data file contains only corpus_id and line_number
    data_file.seek(0)

    parsed_corpus_id = struct.unpack("<B", data_file.read(1))[0]
    assert parsed_corpus_id == corpus_id

    parsed_line_number = struct.unpack("<I", data_file.read(4))[0]
    assert parsed_line_number == original_line_number

    # Should be at end of file
    remaining = data_file.read()
    assert len(remaining) == 0

    # Verify index file
    index_file.seek(0)
    entry_start_pos, src_token_count = struct.unpack("<IB", index_file.read(5))
    assert entry_start_pos == 0
    assert src_token_count == 0  # No source tokens


def test_append_to_dataset_max_source_tokens() -> None:
    """Test with maximum allowed source tokens (255)."""
    data_file = io.BytesIO()
    index_file = io.BytesIO()

    corpus_id = 1
    original_line_number = 1
    src_tokens = list(range(255))  # Exactly 255 tokens
    tgt_tokens = [1000]

    # Should work with 255 tokens
    append_to_dataset(
        data_file, index_file, corpus_id, original_line_number, src_tokens, tgt_tokens
    )

    # Verify the source token count was stored correctly
    index_file.seek(0)
    entry_start_pos, src_token_count = struct.unpack("<IB", index_file.read(5))
    assert src_token_count == 255


def test_append_to_dataset_too_many_source_tokens() -> None:
    """Test that too many source tokens raises an error."""
    data_file = io.BytesIO()
    index_file = io.BytesIO()

    corpus_id = 1
    original_line_number = 1
    src_tokens = list(range(256))  # 256 tokens - should fail
    tgt_tokens = [1000]

    # Should raise ValueError
    try:
        append_to_dataset(
            data_file, index_file, corpus_id, original_line_number, src_tokens, tgt_tokens
        )
        assert False, "Expected ValueError for too many source tokens"
    except ValueError as e:
        assert "exceeds maximum of 255" in str(e)


def test_append_to_dataset_multiple_entries() -> None:
    """Test appending multiple entries sequentially."""
    data_file = io.BytesIO()
    index_file = io.BytesIO()

    # First entry
    append_to_dataset(data_file, index_file, 1, 10, [100], [200])

    # Second entry
    append_to_dataset(data_file, index_file, 2, 20, [300, 400], [500])

    # Verify index file has two entries
    index_file.seek(0)
    index_content = index_file.getvalue()
    assert len(index_content) == 10  # Two entries, 5 bytes each

    # Parse both index entries
    first_entry = struct.unpack("<IB", index_content[:5])
    second_entry = struct.unpack("<IB", index_content[5:])

    # First entry should start at position 0
    assert first_entry[0] == 0
    assert first_entry[1] == 1  # One source token

    # Second entry should start after first entry data
    # First entry: corpus_id(1) + line_number(4) + src_token(2) + tgt_token(2) = 9 bytes
    assert second_entry[0] == 9
    assert second_entry[1] == 2  # Two source tokens


def test_combine_datasets_basic() -> None:
    """Test basic functionality of combine_datasets with temporary files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create two input datasets
        input1_path = os.path.join(tmp_dir, "dataset1")
        input2_path = os.path.join(tmp_dir, "dataset2")
        output_path = os.path.join(tmp_dir, "combined")

        # Create first dataset
        with open(input1_path + ".bin", "wb") as data_file, open(
            input1_path + ".idx", "wb"
        ) as index_file:
            # Add one entry to dataset1
            append_to_dataset(data_file, index_file, 1, 100, [10, 20], [30])

        # Create second dataset
        with open(input2_path + ".bin", "wb") as data_file, open(
            input2_path + ".idx", "wb"
        ) as index_file:
            # Add two entries to dataset2
            append_to_dataset(data_file, index_file, 2, 200, [40], [50, 60])
            append_to_dataset(data_file, index_file, 3, 300, [70, 80, 90], [])

        # Combine datasets
        total_entries = combine_datasets(output_path, [input1_path, input2_path])

        # Verify total entries
        assert total_entries == 3

        # Verify output files exist
        assert os.path.exists(output_path + ".bin")
        assert os.path.exists(output_path + ".idx")

        # Verify index file has correct size (3 entries * 5 bytes each)
        index_size = os.path.getsize(output_path + ".idx")
        assert index_size == 15

        # Verify data file structure by parsing the combined data
        with open(output_path + ".bin", "rb") as data_file, open(
            output_path + ".idx", "rb"
        ) as index_file:

            # Read all index entries
            index_entries = []
            while True:
                entry_data = index_file.read(5)  # Now 5 bytes per entry
                if not entry_data:
                    break
                entry_pos, src_token_count = struct.unpack("<IB", entry_data)
                index_entries.append((entry_pos, src_token_count))

            assert len(index_entries) == 3

            # Verify first entry (from dataset1)
            data_file.seek(index_entries[0][0])
            corpus_id = struct.unpack("<B", data_file.read(1))[0]
            line_num = struct.unpack("<I", data_file.read(4))[0]
            assert corpus_id == 1
            assert line_num == 100
            assert index_entries[0][1] == 2  # Two source tokens

            # Verify second entry (from dataset2)
            data_file.seek(index_entries[1][0])
            corpus_id = struct.unpack("<B", data_file.read(1))[0]
            line_num = struct.unpack("<I", data_file.read(4))[0]
            assert corpus_id == 2
            assert line_num == 200
            assert index_entries[1][1] == 1  # One source token

            # Verify third entry (from dataset2)
            data_file.seek(index_entries[2][0])
            corpus_id = struct.unpack("<B", data_file.read(1))[0]
            line_num = struct.unpack("<I", data_file.read(4))[0]
            assert corpus_id == 3
            assert line_num == 300
            assert index_entries[2][1] == 3  # Three source tokens


def test_combine_datasets_empty_input() -> None:
    """Test combine_datasets with empty input list."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "empty_output")

        # Combine with empty list
        total_entries = combine_datasets(output_path, [])

        # Should return 0 and not create files
        assert total_entries == 0
        assert not os.path.exists(output_path + ".bin")
        assert not os.path.exists(output_path + ".idx")


def test_combine_datasets_missing_files() -> None:
    """Test combine_datasets with non-existent input files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        nonexistent_path = os.path.join(tmp_dir, "nonexistent")
        output_path = os.path.join(tmp_dir, "output")

        # Try to combine non-existent files
        total_entries = combine_datasets(output_path, [nonexistent_path])

        # Should return 0 and not create files
        assert total_entries == 0
        assert not os.path.exists(output_path + ".bin")
        assert not os.path.exists(output_path + ".idx")


def test_combine_datasets_offset_calculation() -> None:
    """Test that offsets are correctly updated when combining datasets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create two datasets with known sizes
        input1_path = os.path.join(tmp_dir, "dataset1")
        input2_path = os.path.join(tmp_dir, "dataset2")
        output_path = os.path.join(tmp_dir, "combined")

        # Create first dataset with specific tokens to control size
        with open(input1_path + ".bin", "wb") as data_file, open(
            input1_path + ".idx", "wb"
        ) as index_file:
            append_to_dataset(data_file, index_file, 1, 1, [100], [200])

        # Get size of first dataset
        first_data_size = os.path.getsize(input1_path + ".bin")

        # Create second dataset
        with open(input2_path + ".bin", "wb") as data_file, open(
            input2_path + ".idx", "wb"
        ) as index_file:
            append_to_dataset(data_file, index_file, 2, 2, [300], [400])

        # Combine datasets
        combine_datasets(output_path, [input1_path, input2_path])

        # Read the combined index file
        with open(output_path + ".idx", "rb") as index_file:
            # First entry offsets should be unchanged (start at 0)
            first_entry_pos, first_src_count = struct.unpack("<IB", index_file.read(5))
            assert first_entry_pos == 0
            assert first_src_count == 1  # One source token

            # Second entry offsets should be shifted by the size of first dataset
            second_entry_pos, second_src_count = struct.unpack("<IB", index_file.read(5))
            assert second_entry_pos == first_data_size
            assert second_src_count == 1  # One source token


def test_get_entry_info_from_index_basic() -> None:
    """Test basic functionality of get_entry_info_from_index."""
    # Create in-memory files with known entries
    data_file = io.BytesIO()
    index_file = io.BytesIO()

    # Add test entries with known token counts
    append_to_dataset(data_file, index_file, 1, 100, [10, 20], [30, 40, 50])  # src=2, tgt=3
    append_to_dataset(data_file, index_file, 2, 200, [60, 70, 80, 90], [100])  # src=4, tgt=1
    append_to_dataset(data_file, index_file, 3, 300, [110] * 10, [120] * 15)  # src=10, tgt=15

    data_file_size = len(data_file.getvalue())

    # Test first entry
    start_pos, src_len, tgt_len = get_entry_info_from_index(index_file, data_file_size, 0)
    assert start_pos == 0
    assert src_len == 2
    assert tgt_len == 3

    # Test second entry
    start_pos, src_len, tgt_len = get_entry_info_from_index(index_file, data_file_size, 1)
    expected_start_pos = 1 + 4 + (2 * 2) + (3 * 2)  # corpus_id + line_num + src_tokens + tgt_tokens
    assert start_pos == expected_start_pos
    assert src_len == 4
    assert tgt_len == 1

    # Test third entry (last entry)
    start_pos, src_len, tgt_len = get_entry_info_from_index(index_file, data_file_size, 2)
    expected_start_pos_2 = expected_start_pos + 1 + 4 + (4 * 2) + (1 * 2)
    assert start_pos == expected_start_pos_2
    assert src_len == 10
    assert tgt_len == 15


def test_get_entry_info_from_index_single_entry() -> None:
    """Test get_entry_info_from_index with a single entry (last entry case)."""
    data_file = io.BytesIO()
    index_file = io.BytesIO()

    # Single entry
    append_to_dataset(data_file, index_file, 1, 42, [100, 200, 300], [400, 500])

    data_file_size = len(data_file.getvalue())

    # Test the single entry (should handle last entry case)
    start_pos, src_len, tgt_len = get_entry_info_from_index(index_file, data_file_size, 0)
    assert start_pos == 0
    assert src_len == 3
    assert tgt_len == 2


def test_get_entry_info_from_index_empty_tokens() -> None:
    """Test get_entry_info_from_index with entries containing empty token lists."""
    data_file = io.BytesIO()
    index_file = io.BytesIO()

    # Entry with no source tokens
    append_to_dataset(data_file, index_file, 1, 10, [], [100, 200])
    # Entry with no target tokens
    append_to_dataset(data_file, index_file, 2, 20, [300, 400], [])
    # Entry with both empty
    append_to_dataset(data_file, index_file, 3, 30, [], [])

    data_file_size = len(data_file.getvalue())

    # Test first entry (no src tokens)
    start_pos, src_len, tgt_len = get_entry_info_from_index(index_file, data_file_size, 0)
    assert start_pos == 0
    assert src_len == 0
    assert tgt_len == 2

    # Test second entry (no tgt tokens)
    start_pos, src_len, tgt_len = get_entry_info_from_index(index_file, data_file_size, 1)
    expected_start_pos = 1 + 4 + 0 + (2 * 2)  # First entry size
    assert start_pos == expected_start_pos
    assert src_len == 2
    assert tgt_len == 0

    # Test third entry (both empty)
    start_pos, src_len, tgt_len = get_entry_info_from_index(index_file, data_file_size, 2)
    expected_start_pos_2 = expected_start_pos + 1 + 4 + (2 * 2) + 0
    assert start_pos == expected_start_pos_2
    assert src_len == 0
    assert tgt_len == 0


def test_get_entry_info_from_index_invalid_entry() -> None:
    """Test get_entry_info_from_index with invalid entry index."""
    data_file = io.BytesIO()
    index_file = io.BytesIO()

    # Add one entry
    append_to_dataset(data_file, index_file, 1, 100, [10], [20])

    data_file_size = len(data_file.getvalue())

    # Try to access non-existent entry
    try:
        get_entry_info_from_index(index_file, data_file_size, 1)  # Entry 1 doesn't exist
        assert False, "Expected ValueError for invalid entry index"
    except ValueError as e:
        assert "Invalid entry index" in str(e)


def test_get_entry_info_from_index_with_files() -> None:
    """Test get_entry_info_from_index using actual files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test_dataset")

        # Create dataset with multiple entries
        with open(dataset_path + ".bin", "wb") as data_file, open(
            dataset_path + ".idx", "wb"
        ) as index_file:
            append_to_dataset(data_file, index_file, 1, 100, [10, 20], [30, 40, 50])
            append_to_dataset(data_file, index_file, 2, 200, [60, 70, 80], [90])
            append_to_dataset(data_file, index_file, 3, 300, [110], [120, 130, 140, 150])

        # Get file sizes
        data_file_size = os.path.getsize(dataset_path + ".bin")

        # Test reading entries from actual files
        with open(dataset_path + ".idx", "rb") as index_file:
            # Test each entry
            start_pos, src_len, tgt_len = get_entry_info_from_index(index_file, data_file_size, 0)
            assert src_len == 2
            assert tgt_len == 3

            start_pos, src_len, tgt_len = get_entry_info_from_index(index_file, data_file_size, 1)
            assert src_len == 3
            assert tgt_len == 1

            start_pos, src_len, tgt_len = get_entry_info_from_index(index_file, data_file_size, 2)
            assert src_len == 1
            assert tgt_len == 4


def test_get_entry_info_from_index_verify_with_data_read() -> None:
    """Test that get_entry_info_from_index info can be used to correctly read data."""
    data_file = io.BytesIO()
    index_file = io.BytesIO()

    # Known test data
    test_entries = [
        (1, 100, [10, 20], [30, 40, 50]),
        (2, 200, [60, 70, 80, 90], [100]),
        (3, 300, [110] * 5, [120] * 8),
    ]

    # Add entries
    for corpus_id, line_num, src_tokens, tgt_tokens in test_entries:
        append_to_dataset(data_file, index_file, corpus_id, line_num, src_tokens, tgt_tokens)

    data_file_size = len(data_file.getvalue())

    # Test each entry by using the info to read data directly
    for i, entry in enumerate(test_entries):
        expected_corpus_id, expected_line_num, expected_src, expected_tgt = entry
        # Get entry info
        start_pos, src_len, tgt_len = get_entry_info_from_index(index_file, data_file_size, i)

        # Verify token lengths match expected
        assert src_len == len(expected_src)
        assert tgt_len == len(expected_tgt)

        # Use the info to read the actual entry data
        data_file.seek(start_pos)

        # Read corpus_id and line_number
        corpus_id = struct.unpack("<B", data_file.read(1))[0]
        line_number = struct.unpack("<I", data_file.read(4))[0]

        assert corpus_id == expected_corpus_id
        assert line_number == expected_line_num

        # Read source tokens
        src_tokens = []
        for _ in range(src_len):
            token = struct.unpack("<H", data_file.read(2))[0]
            src_tokens.append(token)
        assert src_tokens == expected_src

        # Read target tokens
        tgt_tokens = []
        for _ in range(tgt_len):
            token = struct.unpack("<H", data_file.read(2))[0]
            tgt_tokens.append(token)
        assert tgt_tokens == expected_tgt


def test_get_entry_info_from_index_large_entries() -> None:
    """Test get_entry_info_from_index with maximum-sized entries."""
    data_file = io.BytesIO()
    index_file = io.BytesIO()

    # Create entry with maximum source tokens (255)
    max_src_tokens = list(range(255))
    large_tgt_tokens = list(range(1000, 1500))  # 500 target tokens

    append_to_dataset(data_file, index_file, 1, 1, max_src_tokens, large_tgt_tokens)

    data_file_size = len(data_file.getvalue())

    # Test the large entry
    start_pos, src_len, tgt_len = get_entry_info_from_index(index_file, data_file_size, 0)
    assert start_pos == 0
    assert src_len == 255
    assert tgt_len == 500


def test_get_entry_idx_from_bucket_basic() -> None:
    """Test basic functionality of get_entry_idx_from_bucket."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test")

        # Create test dataset with entries of different lengths for different buckets
        with open(dataset_path + ".bin", "wb") as data_file, open(
            dataset_path + ".idx", "wb"
        ) as index_file:
            # These entries should go into different buckets (step_size=16)
            append_to_dataset(
                data_file, index_file, 1, 100, [10] * 5, [20] * 8
            )  # max=8 -> bucket 0
            append_to_dataset(
                data_file, index_file, 2, 200, [30] * 10, [40] * 12
            )  # max=12 -> bucket 0
            append_to_dataset(
                data_file, index_file, 3, 300, [50] * 20, [60] * 18
            )  # max=20 -> bucket 1
            append_to_dataset(
                data_file, index_file, 4, 400, [70] * 25, [80] * 30
            )  # max=30 -> bucket 1
            append_to_dataset(
                data_file, index_file, 5, 500, [90] * 35, [100] * 40
            )  # max=40 -> bucket 2

        # Create chunked index
        create_chunked_index(dataset_path, step_size=16, max_length=64)

        # Test reading from buckets
        with open(dataset_path + ".cidx", "rb") as chunked_file:
            # Bucket 0 should have entries 0 and 1
            entry_idx = get_entry_idx_from_bucket(chunked_file, 0, 0)
            assert entry_idx == 0

            entry_idx = get_entry_idx_from_bucket(chunked_file, 0, 1)
            assert entry_idx == 1

            # Bucket 1 should have entries 2 and 3
            entry_idx = get_entry_idx_from_bucket(chunked_file, 1, 0)
            assert entry_idx == 2

            entry_idx = get_entry_idx_from_bucket(chunked_file, 1, 1)
            assert entry_idx == 3

            # Bucket 2 should have entry 4
            entry_idx = get_entry_idx_from_bucket(chunked_file, 2, 0)
            assert entry_idx == 4


def test_get_entry_idx_from_bucket_error_cases() -> None:
    """Test error handling in get_entry_idx_from_bucket."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test")

        # Create simple test dataset
        with open(dataset_path + ".bin", "wb") as data_file, open(
            dataset_path + ".idx", "wb"
        ) as index_file:
            append_to_dataset(data_file, index_file, 1, 100, [10], [20])
            append_to_dataset(data_file, index_file, 2, 200, [30], [40])

        create_chunked_index(dataset_path, step_size=16, max_length=32)

        with open(dataset_path + ".cidx", "rb") as chunked_file:
            # Test invalid bucket ID
            try:
                get_entry_idx_from_bucket(chunked_file, 10, 0)  # Bucket 10 doesn't exist
                assert False, "Expected ValueError for invalid bucket ID"
            except ValueError as e:
                assert "Bucket ID" in str(e)

            # Test invalid index in bucket
            try:
                get_entry_idx_from_bucket(chunked_file, 0, 10)  # Index 10 doesn't exist in bucket
                assert False, "Expected ValueError for invalid index in bucket"
            except ValueError as e:
                assert "Could not read entry" in str(e)


def test_get_entry_idx_from_bucket_single_bucket() -> None:
    """Test get_entry_idx_from_bucket with all entries in one bucket."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test")

        # Create dataset where all entries fit in bucket 0
        with open(dataset_path + ".bin", "wb") as data_file, open(
            dataset_path + ".idx", "wb"
        ) as index_file:
            append_to_dataset(data_file, index_file, 1, 100, [10] * 3, [20] * 5)  # max=5
            append_to_dataset(data_file, index_file, 2, 200, [30] * 8, [40] * 2)  # max=8
            append_to_dataset(data_file, index_file, 3, 300, [50] * 4, [60] * 6)  # max=6

        create_chunked_index(dataset_path, step_size=16, max_length=32)

        with open(dataset_path + ".cidx", "rb") as chunked_file:
            # All entries should be in bucket 0
            assert get_entry_idx_from_bucket(chunked_file, 0, 0) == 0
            assert get_entry_idx_from_bucket(chunked_file, 0, 1) == 1
            assert get_entry_idx_from_bucket(chunked_file, 0, 2) == 2


def test_get_entry_idx_from_bucket_empty_buckets() -> None:
    """Test get_entry_idx_from_bucket with some empty buckets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test")

        # Create dataset with entries that skip some buckets
        with open(dataset_path + ".bin", "wb") as data_file, open(
            dataset_path + ".idx", "wb"
        ) as index_file:
            # max=8 -> bucket 0
            append_to_dataset(data_file, index_file, 1, 100, [10] * 5, [20] * 8)
            # max=50 -> bucket 3 (skipping buckets 1 and 2)
            append_to_dataset(data_file, index_file, 2, 200, [30] * 50, [40] * 45)

        create_chunked_index(dataset_path, step_size=16, max_length=64)

        with open(dataset_path + ".cidx", "rb") as chunked_file:
            # Bucket 0 should have entry 0
            assert get_entry_idx_from_bucket(chunked_file, 0, 0) == 0

            # Bucket 3 should have entry 1
            assert get_entry_idx_from_bucket(chunked_file, 3, 0) == 1

            # Test reading way out of bounds should fail
            try:
                get_entry_idx_from_bucket(chunked_file, 0, 10)  # Way beyond bucket size
                assert False, "Expected ValueError for way out of range index"
            except ValueError as e:
                assert "Could not read entry" in str(e)


def test_get_entry_idx_from_bucket_large_dataset() -> None:
    """Test get_entry_idx_from_bucket with a larger dataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test")

        # Create dataset with many entries across multiple buckets
        expected_entries = []
        with open(dataset_path + ".bin", "wb") as data_file, open(
            dataset_path + ".idx", "wb"
        ) as index_file:
            for i in range(20):
                # Vary the token lengths to distribute across buckets
                src_len = (i % 5) + 5  # 5-9 tokens
                tgt_len = ((i // 5) * 10) + 10  # 10, 20, 30, 40 tokens

                append_to_dataset(
                    data_file,
                    index_file,
                    i + 1,
                    i * 100,
                    [100 + i] * src_len,
                    [200 + i] * (tgt_len - 1),  # subtract 1 for EOS
                )
                expected_entries.append((i, max(src_len, tgt_len - 1)))

        create_chunked_index(dataset_path, step_size=16, max_length=64)

        # Group expected entries by bucket
        buckets: dict[int, list[int]] = {}
        for entry_idx, max_len in expected_entries:
            bucket_id = min((max_len + 15) // 16 - 1, 3)  # Same logic as in create_chunked_index
            if bucket_id < 0:
                bucket_id = 0
            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append(entry_idx)

        # Test that we can retrieve the correct entries from each bucket
        with open(dataset_path + ".cidx", "rb") as chunked_file:
            for bucket_id, expected_entry_indices in buckets.items():
                for idx_in_bucket, expected_entry_idx in enumerate(expected_entry_indices):
                    actual_entry_idx = get_entry_idx_from_bucket(
                        chunked_file, bucket_id, idx_in_bucket
                    )
                    assert actual_entry_idx == expected_entry_idx, (
                        f"Bucket {bucket_id}, index {idx_in_bucket}: expected {expected_entry_idx},"
                        f" got {actual_entry_idx}"
                    )
