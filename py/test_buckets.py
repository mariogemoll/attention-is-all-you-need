import os
import tempfile

from buckets import (
    create_bucket_index,
    get_bucket_sizes,
    get_entry_idx_from_bucket,
    read_bucket_index_header,
)
from dataset_test_helpers import make_toy_dataset


def test_bucket_index_and_lookup() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "toy")
        # 6 entries, with src/tgt lengths designed to hit different buckets
        entries = [
            ([1, 2], [3, 4]),
            ([1] * 10, [2] * 10),
            ([1] * 20, [2] * 5),
            ([1] * 30, [2] * 30),
            ([1] * 40, [2] * 1),
            ([1] * 50, [2] * 50),
        ]
        make_toy_dataset(prefix, entries)
        # Create bucket index with step_size=16, max_length=64, 1 process
        create_bucket_index(prefix, step_size=16, max_length=64, num_processes=1)
        with open(prefix + ".bidx", "rb") as bidx:
            step, nb, offsets = read_bucket_index_header(bidx)
            assert step == 16
            assert nb == 4
            sizes = get_bucket_sizes(bidx)
            # Check that all entries are assigned to the correct bucket
            # Buckets: 1-16, 17-32, 33-48, 49-64
            assert sizes == [2, 2, 1, 1]
            # Check entry indices in each bucket
            for bucket, expected_idxs in enumerate([[0, 1], [2, 3], [4], [5]]):
                for idx_in_bucket, expected_entry in enumerate(expected_idxs):
                    bidx.seek(0)
                    found = get_entry_idx_from_bucket(bidx, bucket, idx_in_bucket)
                    assert found == expected_entry
