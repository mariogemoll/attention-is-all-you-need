import os
import tempfile
from random import Random
from typing import Sequence

import pytest

from batching import BucketEntries, EpochBatches, get_proc_batches, get_subseq
from buckets import create_bucket_index, open_buckets
from dataset_test_helpers import make_toy_dataset


class BatchingTestData:
    buckets: list[list[int]]
    batch_sizes: list[int]
    num_procs: int

    def __init__(self, buckets: list[list[int]], batch_sizes: list[int], num_procs: int) -> None:
        self.buckets = buckets
        self.batch_sizes = batch_sizes
        self.num_procs = num_procs


@pytest.fixture
def test_data() -> BatchingTestData:
    """Pytest fixture to provide test data."""
    # Create a set of dummy buckets
    buckets = [list(range(1, 101)), list(range(1, 201)), list(range(1, 401))]

    target_num_tokens_per_batch = 256
    step_size = 16
    batch_sizes = [
        max(1, target_num_tokens_per_batch // step_size * (i + 1)) for i in range(len(buckets))
    ]
    num_procs = 3

    return BatchingTestData(buckets=buckets, batch_sizes=batch_sizes, num_procs=num_procs)


def get_seqs_for_processes(
    batch_sizes: list[int],
    buckets: list[list[int]],
    num_procs: int,
    full_batches_only: bool,
    rng_seed: int = 42,
) -> list[list[tuple[int, list[int]]]]:
    """Helper function to get sequences for all processes."""
    results = []

    for i in range(num_procs):
        rng = Random()
        rng.seed(rng_seed)

        proc_batches = get_proc_batches(
            rng=rng,
            batch_sizes=batch_sizes,
            full_batches_only=full_batches_only,
            num_procs=num_procs,
            proc_id=i,
            buckets=buckets,
        )
        results.append(proc_batches)
    return results


def reconstruct_buckets(
    num_buckets: int, results: list[list[tuple[int, list[int]]]]
) -> list[list[int]]:
    """Helper function to reconstruct buckets from process results."""
    reconstructed_buckets: list[list[int]] = [[] for _ in range(num_buckets)]
    for batch_seq in results:
        for bucket_id, seq in batch_seq:
            reconstructed_buckets[bucket_id].extend(seq)
    return reconstructed_buckets


def get_missing(original_bucket: Sequence[int], reconstructed_bucket: Sequence[int]) -> set[int]:
    """Helper function to find missing elements between original and reconstructed buckets."""
    return set(original_bucket) - set(reconstructed_bucket)


def check_batch_sequences(seqs: list[list[tuple[int, list[int]]]]) -> None:
    """Helper function to verify batch sequences have consistent bucket IDs and sizes."""
    # Bucket ids and batch sizes must be identical across all seqs
    # make sure all sequences have the same length
    length = len(seqs[0])
    for seq in seqs:
        assert len(seq) == length
    for i in range(length):
        bucket_id = seqs[0][i][0]
        batch_size = len(seqs[0][i][1])
        for seq in seqs:
            assert seq[i][0] == bucket_id
            assert len(seq[i][1]) == batch_size


def test_only_full_buckets_false(test_data: BatchingTestData) -> None:
    """Test that when full_batches_only=False, missing elements are minimal."""
    results = get_seqs_for_processes(
        test_data.batch_sizes, test_data.buckets, test_data.num_procs, False
    )
    reconstructed_buckets = reconstruct_buckets(len(test_data.buckets), results)
    check_batch_sequences(results)

    for i in range(len(test_data.buckets)):
        missing = get_missing(test_data.buckets[i], reconstructed_buckets[i])
        assert len(missing) < test_data.num_procs


def test_only_full_buckets_true(test_data: BatchingTestData) -> None:
    """Test that when full_batches_only=True, missing elements are within expected bounds."""
    results = get_seqs_for_processes(
        test_data.batch_sizes, test_data.buckets, test_data.num_procs, True
    )
    reconstructed_buckets = reconstruct_buckets(len(test_data.buckets), results)
    check_batch_sequences(results)

    for i in range(len(test_data.buckets)):
        missing = get_missing(test_data.buckets[i], reconstructed_buckets[i])
        assert len(missing) < test_data.batch_sizes[i] * test_data.num_procs


def test_epoch_batches_basic() -> None:
    """Test basic functionality of EpochBatches with new dataset format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "test")

        # Create test dataset with more entries to ensure non-empty buckets
        entries = [
            ([1, 2], [3, 4]),  # Length 2
            ([5, 6, 7, 8], [9, 10]),  # Length 4
            ([11], [12]),  # Length 1
            ([13, 14, 15], [16, 17, 18]),  # Length 3
            ([19, 20, 21, 22, 23], [24, 25]),  # Length 5
        ]
        make_toy_dataset(base, entries)

        # Create bucket index with smaller step size for better distribution
        create_bucket_index(base, step_size=2, max_length=16, num_processes=1)

        # Test EpochBatches
        with open(base + ".bidx", "rb") as bucket_file:
            epoch_batches = EpochBatches(
                num_procs=1,
                proc_id=0,
                bucket_index_file=bucket_file,
                target_num_tokens_per_batch=8,
                rng_seed=42,
                full_batches_only=False,
            )

            # Should be able to iterate and get batches
            batches = list(epoch_batches)
            assert len(batches) > 0

            # Each batch should be a tuple of (bucket_id, entry_indices)
            for bucket_id, entry_indices in batches:
                assert isinstance(bucket_id, int)
                assert isinstance(entry_indices, list)
                assert all(isinstance(idx, int) for idx in entry_indices)


def test_bucket_entries_basic() -> None:
    """Test basic functionality of BucketEntries with new dataset format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "test")

        # Create test dataset
        entries = [
            ([1, 2, 3], [4, 5]),
            ([6, 7], [8, 9, 10]),
        ]
        make_toy_dataset(base, entries)

        # Create bucket index
        create_bucket_index(base, step_size=4, max_length=16, num_processes=1)

        # Test BucketEntries
        with open_buckets(base) as bucketed_dataset:
            bucket_entries = BucketEntries(bucketed_dataset, bucket_id=0)

            # Should be able to get length
            assert len(bucket_entries) >= 0

            # Should be able to iterate
            entries_list = list(bucket_entries)

            # Each entry should be a tuple of (original_line_number, src_tokens)
            for original_line_number, src_tokens in entries_list:
                assert isinstance(original_line_number, int)
                assert isinstance(src_tokens, list)
                assert all(isinstance(token, int) for token in src_tokens)


def test_get_subseq_single_processor() -> None:
    """Test get_subseq with num_procs=1."""
    rng = Random(42)
    entries = [1, 2, 3, 4, 5]

    # Should return a copy of all entries
    result = get_subseq(rng, num_procs=1, proc_id=0, entries=entries)
    assert result == entries
    assert result is not entries  # Should be a copy

    # Test with empty list
    result_empty: list[int] = get_subseq(rng, num_procs=1, proc_id=0, entries=[])
    assert result_empty == []


def test_get_subseq_single_processor_invalid_proc_id() -> None:
    """Test get_subseq with num_procs=1 and invalid proc_id."""
    rng = Random(42)
    entries = [1, 2, 3, 4, 5]

    # Should raise AssertionError for proc_id != 0 when num_procs=1
    with pytest.raises(AssertionError, match="proc_id must be 0 if num_procs is 1"):
        get_subseq(rng, num_procs=1, proc_id=1, entries=entries)


def test_get_subseq_entries_less_than_procs() -> None:
    """Test get_subseq when number of entries is less than number of processors."""
    rng = Random(42)
    entries = [1, 2]  # Only 2 entries
    num_procs = 4  # But 4 processors

    # Should raise ValueError
    with pytest.raises(
        ValueError, match="Number of entries \\(2\\) is less than number of processors \\(4\\)"
    ):
        get_subseq(rng, num_procs=num_procs, proc_id=0, entries=entries)

    # Test edge case: exactly equal should work
    entries_equal = [1, 2, 3, 4]
    result = get_subseq(rng, num_procs=4, proc_id=0, entries=entries_equal)
    assert len(result) == 1  # Should get 1 entry per processor


def test_get_subseq_empty_entries() -> None:
    """Test get_subseq with empty entries list."""
    rng = Random(42)

    # Empty entries should always return empty list regardless of num_procs
    assert get_subseq(rng, num_procs=1, proc_id=0, entries=[]) == []
    assert get_subseq(rng, num_procs=4, proc_id=2, entries=[]) == []


if __name__ == "__main__":
    pytest.main([__file__])
