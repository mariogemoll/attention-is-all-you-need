import random
from typing import BinaryIO, Generator, TypeVar

from buckets import (
    BucketedDataset,
    get_bucket_sizes,
    get_entry_idx_from_bucket,
    read_bucket_index_header,
)
from dataset import get_entry

T = TypeVar("T")


def get_subseq(rng: random.Random, num_procs: int, proc_id: int, entries: list[T]) -> list[T]:
    # Handle empty buckets gracefully
    if len(entries) == 0:
        return []

    if num_procs == 1:
        assert proc_id == 0, "proc_id must be 0 if num_procs is 1"
        return entries.copy()

    if len(entries) < num_procs:
        raise ValueError(
            f"Number of entries ({len(entries)}) is less than number of processors ({num_procs})"
        )

    entries = entries.copy()
    rng.shuffle(entries)

    # Truncate entries to the nearest multiple of num_procs
    entries = entries[: len(entries) - (len(entries) % num_procs)]

    # Distribute all entries among processors (no wasteful truncation)
    # Each processor gets every num_procs-th element starting from proc_id
    return entries[proc_id::num_procs]


def get_batches(batch_size: int, full_batches_only: bool, entries: list[T]) -> list[list[T]]:
    if len(entries) == 0:
        return []

    batches_list = [entries[i : i + batch_size] for i in range(0, len(entries), batch_size)]
    if full_batches_only and len(batches_list) > 0 and len(batches_list[-1]) != batch_size:
        return batches_list[:-1]
    else:
        return batches_list


def get_proc_batches(
    rng: random.Random,
    batch_sizes: list[int],
    full_batches_only: bool,
    num_procs: int,
    proc_id: int,
    buckets: list[list[T]],
) -> list[tuple[int, list[T]]]:
    assert len(batch_sizes) == len(buckets)

    # Randomize the contents of the buckets and get our subsets, and create batches from those
    my_batches = [
        get_batches(batch_sizes[i], full_batches_only, get_subseq(rng, num_procs, proc_id, bucket))
        for i, bucket in enumerate(buckets)
    ]

    # get the number of batches per bucket
    num_batches_per_bucket = [len(batches) for batches in my_batches]

    bucket_id_seq = [i for i, count in enumerate(num_batches_per_bucket) for _ in range(count)]
    # Shuffle the bucket IDs
    rng.shuffle(bucket_id_seq)

    cursors = [0] * len(my_batches)
    final_batch_seq = []

    for bucket_id in bucket_id_seq:
        if cursors[bucket_id] < len(my_batches[bucket_id]):
            final_batch_seq.append((bucket_id, my_batches[bucket_id][cursors[bucket_id]]))
            cursors[bucket_id] += 1

    return final_batch_seq


class EpochBatches:
    """
    Custom iterable for epoch batches that supports len().

    Args:
        bucket_index_file: Open binary file handle for the bucket index file (.bidx)
        bucket_index_path: Path to the bucket index file (for reading header/sizes)
        target_num_tokens_per_batch: Target number of tokens per batch
        shuffle_within_buckets: Whether to shuffle entries within each bucket
        shuffle_batches: Whether to shuffle the order of batches
        random_seed: Random seed for reproducibility
        full_batches_only: If True, omit the last batches from buckets if they're not full
    """

    def __init__(
        self,
        num_procs: int,
        proc_id: int,
        bucket_index_file: BinaryIO,
        target_num_tokens_per_batch: int,
        rng_seed: int | float | str | bytes | bytearray | None,
        full_batches_only: bool = False,
    ):
        assert proc_id < num_procs
        self.bucket_index_file = bucket_index_file

        header = read_bucket_index_header(self.bucket_index_file)
        step_size, num_buckets, _ = header
        bucket_sizes = get_bucket_sizes(self.bucket_index_file)

        # Calculate batch sizes for the buckets
        buckets = [list(range(size)) for size in bucket_sizes]
        batch_sizes = [
            max(1, target_num_tokens_per_batch // (step_size * (i + 1)))
            for i in range(len(buckets))
        ]
        # Round down to nearest multiple of 16, but ensure minimum of 1
        batch_sizes = [max(1, size - (size % 16)) for size in batch_sizes]

        rng = random.Random(rng_seed)
        self.batches = get_proc_batches(
            rng, batch_sizes, full_batches_only, num_procs, proc_id, buckets
        )

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Generator[tuple[int, list[int]], None, None]:
        # Yield batches one by one, resolving indices on demand
        for bucket_id, idxs in self.batches:
            # # Get the bucket-relative indices for this batch
            # shuffled_entries = self.bucket_shuffled_entries[bucket_id]
            # bucket_batch = shuffled_entries[batch_start_idx : batch_start_idx + batch_size]

            # Resolve bucket indices to actual dataset entry indices
            resolved_batch = []
            for entry_idx_in_bucket in idxs:
                actual_entry_idx = get_entry_idx_from_bucket(
                    self.bucket_index_file, bucket_id, entry_idx_in_bucket
                )
                resolved_batch.append(actual_entry_idx)

            yield (bucket_id, resolved_batch)


class BucketEntries:
    """
    Iterable for all entries in a specific bucket that supports len().

    Args:
        dataset: BucketedDataset instance
        bucket_id: ID of the bucket to iterate over
    """

    def __init__(self, dataset: BucketedDataset, bucket_id: int):
        self.dataset = dataset
        self.bucket_id = bucket_id

        bucket_sizes = get_bucket_sizes(dataset.bucket_index_file)

        if bucket_id >= len(bucket_sizes):
            raise ValueError(
                f"Bucket ID {bucket_id} is out of range (max: {len(bucket_sizes) - 1})"
            )

        self.bucket_size = bucket_sizes[bucket_id]

    def __len__(self) -> int:
        return self.bucket_size

    def __iter__(self) -> Generator[tuple[int, list[int]], None, None]:
        """Yield (original_line_number, src_tokens) for each entry in the bucket."""
        for idx_in_bucket in range(self.bucket_size):
            entry_idx = get_entry_idx_from_bucket(
                self.dataset.bucket_index_file, self.bucket_id, idx_in_bucket
            )
            corpus_id, original_line_number, src_tokens, tgt_tokens = get_entry(
                self.dataset.dataset, entry_idx
            )
            yield (original_line_number, src_tokens)
