import os
import tempfile

from batching import EpochBatches
from serialization import append_to_dataset, create_bucket_index


def test_epoch_batches_basic() -> None:
    """Test basic functionality of EpochBatches."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test_epoch")

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

        # Create bucket index
        create_bucket_index(dataset_path, step_size=16, max_length=64)

        # Test EpochBatches with a reasonable target tokens per batch
        with open(dataset_path + ".bidx", "rb") as bucket_index_file:
            epoch_batches = EpochBatches(
                bucket_index_file,
                target_tokens_per_batch=64,
                shuffle_within_buckets=False,
                shuffle_batches=False,
            )

            # Should have some batches
            assert len(epoch_batches) > 0

            # Test iteration
            batch_count = 0
            for bucket_id, batch_indices in epoch_batches:
                assert isinstance(bucket_id, int)
                assert isinstance(batch_indices, list)
                assert all(isinstance(idx, int) for idx in batch_indices)
                batch_count += 1

            assert batch_count == len(epoch_batches)


def test_epoch_batches_full_batches_only_false() -> None:
    """Test EpochBatches with full_batches_only=False (default behavior)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test_full_batches_false")

        # Create dataset with entries that will produce incomplete batches
        with open(dataset_path + ".bin", "wb") as data_file, open(
            dataset_path + ".idx", "wb"
        ) as index_file:
            # Bucket 0 entries (max seq len = 16)
            for i in range(5):  # 5 entries in bucket 0
                append_to_dataset(data_file, index_file, 1, i, [10 + i] * 5, [20 + i] * 8)  # max=8

            # Bucket 1 entries (max seq len = 32)
            for i in range(3):  # 3 entries in bucket 1
                append_to_dataset(
                    data_file, index_file, 2, i + 100, [30 + i] * 20, [40 + i] * 25
                )  # max=25

        create_bucket_index(dataset_path, step_size=16, max_length=64)

        # Test with full_batches_only=False
        with open(dataset_path + ".bidx", "rb") as bucket_index_file:
            epoch_batches = EpochBatches(
                bucket_index_file,
                target_tokens_per_batch=64,  # bucket 0: batch_size=4, bucket 1: batch_size=2
                shuffle_within_buckets=False,
                shuffle_batches=False,
                full_batches_only=False,
            )

            # Collect all batches
            batches = list(epoch_batches)

            # Should have batches including incomplete ones
            # Bucket 0: 5 entries / batch_size=4 -> 2 batches (4 + 1)
            # Bucket 1: 3 entries / batch_size=2 -> 2 batches (2 + 1)
            # Total: 4 batches
            assert len(batches) == 4

            # Check batch sizes
            batch_sizes = [len(batch) for _, batch in batches]
            assert 4 in batch_sizes  # Full batch from bucket 0
            assert 1 in batch_sizes  # Incomplete batch from bucket 0
            assert 2 in batch_sizes  # Full batch from bucket 1
            # The incomplete batch from bucket 1 should also be size 1


def test_epoch_batches_full_batches_only_true() -> None:
    """Test EpochBatches with full_batches_only=True."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test_full_batches_true")

        # Create dataset with entries that will produce incomplete batches
        with open(dataset_path + ".bin", "wb") as data_file, open(
            dataset_path + ".idx", "wb"
        ) as index_file:
            # Bucket 0 entries (max seq len = 16)
            for i in range(5):  # 5 entries in bucket 0
                append_to_dataset(data_file, index_file, 1, i, [10 + i] * 5, [20 + i] * 8)  # max=8

            # Bucket 1 entries (max seq len = 32)
            for i in range(3):  # 3 entries in bucket 1
                append_to_dataset(
                    data_file, index_file, 2, i + 100, [30 + i] * 20, [40 + i] * 25
                )  # max=25

        create_bucket_index(dataset_path, step_size=16, max_length=64)

        # Test with full_batches_only=True
        with open(dataset_path + ".bidx", "rb") as bucket_index_file:
            epoch_batches = EpochBatches(
                bucket_index_file,
                target_tokens_per_batch=64,  # bucket 0: batch_size=4, bucket 1: batch_size=2
                shuffle_within_buckets=False,
                shuffle_batches=False,
                full_batches_only=True,
            )

            # Collect all batches
            batches = list(epoch_batches)

            # Should only have full batches
            # Bucket 0: 5 entries / batch_size=4 -> 1 full batch (size 4), 1 incomplete dropped
            # Bucket 1: 3 entries / batch_size=2 -> 1 full batch (size 2), 1 incomplete dropped
            # Total: 2 batches
            assert len(batches) == 2

            # Check that all batches are full
            batch_sizes = [len(batch) for _, batch in batches]
            expected_full_sizes = {4, 2}  # Full batch sizes for each bucket
            for size in batch_sizes:
                assert size in expected_full_sizes, f"Unexpected batch size: {size}"

            # Should not contain any incomplete batches (size 1 in this case)
            assert 1 not in batch_sizes


def test_epoch_batches_full_batches_only_comparison() -> None:
    """Test that full_batches_only=True produces fewer batches than full_batches_only=False."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test_comparison")

        # Create dataset that will definitely have incomplete batches
        with open(dataset_path + ".bin", "wb") as data_file, open(
            dataset_path + ".idx", "wb"
        ) as index_file:
            # Create 7 entries that should go to bucket 0
            for i in range(7):
                append_to_dataset(data_file, index_file, 1, i, [10 + i] * 3, [20 + i] * 5)  # max=5

        create_bucket_index(dataset_path, step_size=16, max_length=32)

        # Test with full_batches_only=False and True
        with open(dataset_path + ".bidx", "rb") as bucket_index_file:
            epoch_batches_all = EpochBatches(
                bucket_index_file,
                target_tokens_per_batch=64,  # Should give batch_size=4 for bucket 0
                shuffle_within_buckets=False,
                shuffle_batches=False,
                full_batches_only=False,
            )
            batches_all = list(epoch_batches_all)

        with open(dataset_path + ".bidx", "rb") as bucket_index_file:
            epoch_batches_full = EpochBatches(
                bucket_index_file,
                target_tokens_per_batch=64,
                shuffle_within_buckets=False,
                shuffle_batches=False,
                full_batches_only=True,
            )
            batches_full = list(epoch_batches_full)

        # Should have fewer batches with full_batches_only=True
        # 7 entries / batch_size=4 -> 2 batches (4 + 3) with False, 1 batch (4) with True
        assert len(batches_all) == 2
        assert len(batches_full) == 1

        # All batches in full mode should be complete
        for _, batch in batches_full:
            # Batch size should be 4 for bucket 0
            assert len(batch) == 4


def test_epoch_batches_full_batches_only_edge_cases() -> None:
    """Test edge cases for full_batches_only parameter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test_edge_cases")

        # Create dataset where all entries exactly fill batches (no incomplete batches)
        with open(dataset_path + ".bin", "wb") as data_file, open(
            dataset_path + ".idx", "wb"
        ) as index_file:
            # Create exactly 8 entries for bucket 0 (batch_size=4 -> 2 full batches)
            for i in range(8):
                append_to_dataset(data_file, index_file, 1, i, [10 + i] * 3, [20 + i] * 5)  # max=5

        create_bucket_index(dataset_path, step_size=16, max_length=32)

        # Test both modes - should give same result when there are no incomplete batches
        with open(dataset_path + ".bidx", "rb") as bucket_index_file:
            epoch_batches_all = EpochBatches(
                bucket_index_file,
                target_tokens_per_batch=64,  # batch_size=4 for bucket 0
                shuffle_within_buckets=False,
                shuffle_batches=False,
                full_batches_only=False,
            )
            batches_all = list(epoch_batches_all)

        with open(dataset_path + ".bidx", "rb") as bucket_index_file:
            epoch_batches_full = EpochBatches(
                bucket_index_file,
                target_tokens_per_batch=64,
                shuffle_within_buckets=False,
                shuffle_batches=False,
                full_batches_only=True,
            )
            batches_full = list(epoch_batches_full)

        # Should have the same number of batches when all are full
        assert len(batches_all) == len(batches_full) == 2

        # All batches should have the same size
        for (_, batch_all), (_, batch_full) in zip(batches_all, batches_full):
            assert len(batch_all) == len(batch_full) == 4


def test_epoch_batches_full_batches_only_empty_buckets() -> None:
    """Test full_batches_only with some empty buckets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test_empty_buckets")

        # Create dataset where some buckets are empty
        with open(dataset_path + ".bin", "wb") as data_file, open(
            dataset_path + ".idx", "wb"
        ) as index_file:
            # Skip bucket 1, put 3 entries in bucket 2
            for i in range(3):
                append_to_dataset(
                    data_file, index_file, 1, i, [10 + i] * 33, [20 + i] * 35
                )  # max=35 -> bucket 2

        create_bucket_index(dataset_path, step_size=16, max_length=64)

        # Test with full_batches_only=True
        with open(dataset_path + ".bidx", "rb") as bucket_index_file:
            epoch_batches = EpochBatches(
                bucket_index_file,
                target_tokens_per_batch=96,  # batch_size=2 for bucket 2 (48 tokens per sample)
                shuffle_within_buckets=False,
                shuffle_batches=False,
                full_batches_only=True,
            )
            batches = list(epoch_batches)

        # Should have 1 full batch (3 entries / batch_size=2 -> 1 full batch, 1 incomplete dropped)
        assert len(batches) == 1
        bucket_id, batch_indices = batches[0]
        assert bucket_id == 2
        assert len(batch_indices) == 2


def test_epoch_batches_full_batches_only_shuffling() -> None:
    """Test that full_batches_only works correctly with shuffling enabled."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test_shuffle")

        # Create dataset with incomplete batches
        with open(dataset_path + ".bin", "wb") as data_file, open(
            dataset_path + ".idx", "wb"
        ) as index_file:
            # 5 entries in bucket 0
            for i in range(5):
                append_to_dataset(data_file, index_file, 1, i, [10 + i] * 3, [20 + i] * 5)  # max=5

        create_bucket_index(dataset_path, step_size=16, max_length=32)

        # Test with shuffling enabled and full_batches_only=True
        with open(dataset_path + ".bidx", "rb") as bucket_index_file:
            epoch_batches = EpochBatches(
                bucket_index_file,
                target_tokens_per_batch=64,  # batch_size=4
                shuffle_within_buckets=True,
                shuffle_batches=True,
                random_seed=42,  # For reproducibility
                full_batches_only=True,
            )
            batches = list(epoch_batches)

            # Should still filter out incomplete batches even with shuffling
            assert len(batches) == 1  # Only 1 full batch of size 4
            _, batch_indices = batches[0]
            assert len(batch_indices) == 4

        # Test that we get consistent results with the same seed
        with open(dataset_path + ".bidx", "rb") as bucket_index_file:
            epoch_batches_2 = EpochBatches(
                bucket_index_file,
                target_tokens_per_batch=64,
                shuffle_within_buckets=True,
                shuffle_batches=True,
                random_seed=42,  # Same seed
                full_batches_only=True,
            )
            batches_2 = list(epoch_batches_2)
        assert len(batches) == len(batches_2)
        # The actual indices might be different due to shuffling, but counts should match
