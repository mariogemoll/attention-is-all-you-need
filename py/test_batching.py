from random import Random
from typing import Sequence

import pytest

from batching import get_proc_batches


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


if __name__ == "__main__":
    pytest.main([__file__])
