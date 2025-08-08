import random
from typing import BinaryIO, Generator

from serialization import (
    get_bucket_sizes,
    get_entry_idx_from_bucket,
    read_bucket_index_header,
)


class EpochBatches:
    """
    Custom iterable for epoch batches that supports len().

    Args:
        bucket_index_file: Open binary file handle for the bucket index file (.bidx)
        bucket_index_path: Path to the bucket index file (for reading header/sizes)
        target_tokens_per_batch: Target number of tokens per batch
        shuffle_within_buckets: Whether to shuffle entries within each bucket
        shuffle_batches: Whether to shuffle the order of batches
        random_seed: Random seed for reproducibility
        full_batches_only: If True, omit the last batches from buckets if they're not full
    """

    def __init__(
        self,
        bucket_index_file: BinaryIO,
        target_tokens_per_batch: int,
        shuffle_within_buckets: bool = True,
        shuffle_batches: bool = True,
        random_seed: int | None = None,
        full_batches_only: bool = False,
    ):
        self.bucket_index_file = bucket_index_file
        self.target_tokens_per_batch = target_tokens_per_batch
        self.shuffle_within_buckets = shuffle_within_buckets
        self.shuffle_batches = shuffle_batches
        self.random_seed = random_seed
        self.full_batches_only = full_batches_only

        # Pre-calculate batch schedule and total count
        if self.random_seed is not None:
            random.seed(self.random_seed)

        header = read_bucket_index_header(bucket_index_file)
        self.step_size, self.num_buckets, self.bucket_offsets = header
        self.bucket_sizes = get_bucket_sizes(bucket_index_file)

        # Store shuffled bucket entries
        self.bucket_shuffled_entries = {}
        for bucket_id, size in enumerate(self.bucket_sizes):
            if size > 0:  # Skip empty buckets
                entry_indices_in_bucket = list(range(size))
                if self.shuffle_within_buckets:
                    random.shuffle(entry_indices_in_bucket)
                self.bucket_shuffled_entries[bucket_id] = entry_indices_in_bucket

        # Create batch schedule (bucket_id, start_idx, batch_size)
        self.batch_schedule = []
        for bucket_id, size in enumerate(self.bucket_sizes):
            if size > 0:
                # Calculate batch size for this bucket based on max sequence length
                max_seq_len = (bucket_id + 1) * self.step_size
                batch_size = max(1, self.target_tokens_per_batch // max_seq_len)

                # Add batch schedule entries for this bucket
                for i in range(0, size, batch_size):
                    actual_batch_size = min(batch_size, size - i)

                    # Skip incomplete batches if full_batches_only is True
                    if self.full_batches_only and actual_batch_size < batch_size:
                        continue

                    self.batch_schedule.append((bucket_id, i, actual_batch_size))

        # Shuffle the batch schedule
        if self.shuffle_batches:
            random.shuffle(self.batch_schedule)

    def __len__(self) -> int:
        return len(self.batch_schedule)

    def __iter__(self) -> Generator[tuple[int, list[int]], None, None]:
        # Yield batches one by one, resolving indices on demand
        for bucket_id, batch_start_idx, batch_size in self.batch_schedule:
            # Get the bucket-relative indices for this batch
            shuffled_entries = self.bucket_shuffled_entries[bucket_id]
            bucket_batch = shuffled_entries[batch_start_idx : batch_start_idx + batch_size]

            # Resolve bucket indices to actual dataset entry indices
            resolved_batch = []
            for entry_idx_in_bucket in bucket_batch:
                actual_entry_idx = get_entry_idx_from_bucket(
                    self.bucket_index_file, bucket_id, entry_idx_in_bucket
                )
                resolved_batch.append(actual_entry_idx)

            yield (bucket_id, resolved_batch)
