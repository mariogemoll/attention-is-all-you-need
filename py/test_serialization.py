import io
import os
import struct
import tempfile

from serialization import append_to_dataset, combine_datasets


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
