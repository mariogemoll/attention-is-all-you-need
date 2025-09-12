import os
import tempfile

import torch

from dataset import open_dataset
from dataset_test_helpers import make_toy_dataset
from params import eos, pad, sos
from tensors import get_tensors


def test_get_tensors_basic() -> None:
    """Test basic functionality of get_tensors with simple data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "test")

        # Create test dataset with known data
        entries = [
            ([10, 20], [30, 40]),
            ([50, 60, 70], [80, 90]),
        ]
        make_toy_dataset(base, entries)

        with open_dataset(base) as dataset:
            seq_len = 5
            entry_indices = [0, 1]

            enc_input, dec_input, dec_target = get_tensors(seq_len, dataset, entry_indices)

            # Expected encoder input: src_tokens + padding
            expected_enc_input = [
                [10, 20, pad, pad, pad],  # [10, 20] + 3 padding
                [50, 60, 70, pad, pad],  # [50, 60, 70] + 2 padding
            ]

            # Expected decoder input: [sos] + tgt_tokens + padding
            expected_dec_input = [
                [sos, 30, 40, pad, pad],  # [sos] + [30, 40] + 2 padding
                [sos, 80, 90, pad, pad],  # [sos] + [80, 90] + 2 padding
            ]

            # Expected decoder target: tgt_tokens + [eos] + padding
            expected_dec_target = [
                [30, 40, eos, pad, pad],  # [30, 40] + [eos] + 2 padding
                [80, 90, eos, pad, pad],  # [80, 90] + [eos] + 2 padding
            ]

            # Verify tensor shapes
            assert enc_input.shape == (2, 5)
            assert dec_input.shape == (2, 5)
            assert dec_target.shape == (2, 5)

            # Verify tensor values
            assert torch.equal(enc_input, torch.tensor(expected_enc_input, dtype=torch.int64))
            assert torch.equal(dec_input, torch.tensor(expected_dec_input, dtype=torch.int64))
            assert torch.equal(dec_target, torch.tensor(expected_dec_target, dtype=torch.int64))


def test_get_tensors_exact_length() -> None:
    """Test when token sequences exactly match the sequence length."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "test")

        entries = [([10, 20, 30], [40, 50])]  # src_len=3, tgt_len=2, seq_len=4
        make_toy_dataset(base, entries)

        with open_dataset(base) as dataset:
            seq_len = 4
            entry_indices = [0]

            enc_input, dec_input, dec_target = get_tensors(seq_len, dataset, entry_indices)

            # Expected: src_tokens + 1 padding (seq_len=4, src_len=3)
            expected_enc_input = [[10, 20, 30, pad]]

            # Expected: [sos] + tgt_tokens + 1 padding (seq_len=4, tgt_len=2, +sos=3)
            expected_dec_input = [[sos, 40, 50, pad]]

            # Expected: tgt_tokens + [eos] + 1 padding (seq_len=4, tgt_len=2, +eos=3)
            expected_dec_target = [[40, 50, eos, pad]]

            assert torch.equal(enc_input, torch.tensor(expected_enc_input, dtype=torch.int64))
            assert torch.equal(dec_input, torch.tensor(expected_dec_input, dtype=torch.int64))
            assert torch.equal(dec_target, torch.tensor(expected_dec_target, dtype=torch.int64))


def test_get_tensors_empty_sequences() -> None:
    """Test handling of empty token sequences."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "test")

        entries: list[tuple[list[int], list[int]]] = [([], [])]  # Empty source and target
        make_toy_dataset(base, entries)

        with open_dataset(base) as dataset:
            seq_len = 3
            entry_indices = [0]

            enc_input, dec_input, dec_target = get_tensors(seq_len, dataset, entry_indices)

            # Expected: all padding for encoder input
            expected_enc_input = [[pad, pad, pad]]

            # Expected: [sos] + 2 padding for decoder input
            expected_dec_input = [[sos, pad, pad]]

            # Expected: [eos] + 2 padding for decoder target
            expected_dec_target = [[eos, pad, pad]]

            assert torch.equal(enc_input, torch.tensor(expected_enc_input, dtype=torch.int64))
            assert torch.equal(dec_input, torch.tensor(expected_dec_input, dtype=torch.int64))
            assert torch.equal(dec_target, torch.tensor(expected_dec_target, dtype=torch.int64))


def test_get_tensors_single_entry() -> None:
    """Test with single entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "test")

        entries = [([100], [200, 300])]
        make_toy_dataset(base, entries)

        with open_dataset(base) as dataset:
            seq_len = 4
            entry_indices = [0]

            enc_input, dec_input, dec_target = get_tensors(seq_len, dataset, entry_indices)

            # Verify batch dimension is 1
            assert enc_input.shape == (1, 4)
            assert dec_input.shape == (1, 4)
            assert dec_target.shape == (1, 4)

            expected_enc_input = [[100, pad, pad, pad]]
            expected_dec_input = [[sos, 200, 300, pad]]
            expected_dec_target = [[200, 300, eos, pad]]

            assert torch.equal(enc_input, torch.tensor(expected_enc_input, dtype=torch.int64))
            assert torch.equal(dec_input, torch.tensor(expected_dec_input, dtype=torch.int64))
            assert torch.equal(dec_target, torch.tensor(expected_dec_target, dtype=torch.int64))


def test_get_tensors_tensor_dtype() -> None:
    """Test that returned tensors have correct dtype."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "test")

        entries = [([1, 2], [3, 4])]
        make_toy_dataset(base, entries)

        with open_dataset(base) as dataset:
            seq_len = 3
            entry_indices = [0]

            enc_input, dec_input, dec_target = get_tensors(seq_len, dataset, entry_indices)

            assert enc_input.dtype == torch.int64
            assert dec_input.dtype == torch.int64
            assert dec_target.dtype == torch.int64


def test_get_tensors_multiple_entries_different_lengths() -> None:
    """Test with multiple entries having different token lengths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "test")

        entries = [
            ([1], [2, 3, 4]),  # Short src, long tgt
            ([5, 6, 7, 8], [9]),  # Long src, short tgt
            ([10, 11], [12, 13]),  # Medium lengths
        ]
        make_toy_dataset(base, entries)

        with open_dataset(base) as dataset:
            seq_len = 6
            entry_indices = [0, 1, 2]

            enc_input, dec_input, dec_target = get_tensors(seq_len, dataset, entry_indices)

            # Verify shape
            assert enc_input.shape == (3, 6)
            assert dec_input.shape == (3, 6)
            assert dec_target.shape == (3, 6)

            expected_enc_input = [
                [1, pad, pad, pad, pad, pad],  # [1] + 5 padding
                [5, 6, 7, 8, pad, pad],  # [5,6,7,8] + 2 padding
                [10, 11, pad, pad, pad, pad],  # [10,11] + 4 padding
            ]

            expected_dec_input = [
                [sos, 2, 3, 4, pad, pad],  # [sos] + [2,3,4] + 2 padding
                [sos, 9, pad, pad, pad, pad],  # [sos] + [9] + 4 padding
                [sos, 12, 13, pad, pad, pad],  # [sos] + [12,13] + 3 padding
            ]

            expected_dec_target = [
                [2, 3, 4, eos, pad, pad],  # [2,3,4] + [eos] + 2 padding
                [9, eos, pad, pad, pad, pad],  # [9] + [eos] + 4 padding
                [12, 13, eos, pad, pad, pad],  # [12,13] + [eos] + 3 padding
            ]

            assert torch.equal(enc_input, torch.tensor(expected_enc_input, dtype=torch.int64))
            assert torch.equal(dec_input, torch.tensor(expected_dec_input, dtype=torch.int64))
            assert torch.equal(dec_target, torch.tensor(expected_dec_target, dtype=torch.int64))
