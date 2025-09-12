import sys

from tqdm import tqdm

from batching import EpochBatches
from buckets import open_buckets, read_bucket_index_header
from dataset import get_entry


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <file_path_prefix>")
        sys.exit(1)

    batch_counter = 0
    pair_counter = 0

    file_path_prefix = sys.argv[1]

    target_num_tokens_per_batch = 32768

    with open_buckets(file_path_prefix) as buckets_ds:
        step_size, _, _ = read_bucket_index_header(buckets_ds.bucket_index_file)
        batches = EpochBatches(
            num_procs=1,
            proc_id=0,
            bucket_index_file=buckets_ds.bucket_index_file,
            target_num_tokens_per_batch=target_num_tokens_per_batch,
            rng_seed=None,
            full_batches_only=True,
        )

        print(f"Total number of batches: {len(batches)}")

        for bucket_id, batch_indices in tqdm(batches):

            prev_bucket_seq_len = bucket_id * step_size
            bucket_seq_len = (bucket_id + 1) * step_size

            for actual_entry_idx in batch_indices:

                batch_counter += 1
                _, _, src_tokens, tgt_tokens = get_entry(buckets_ds.dataset, actual_entry_idx)
                pair_counter += 1

                src_len = len(src_tokens)
                tgt_len = len(tgt_tokens) + 1  # +1 for the start/end token

                # Make sure none of the lengths is bigger than the bucket size
                assert (
                    src_len <= bucket_seq_len
                ), f"Error: Source tokens len {src_len} exceeds bucket seq len {bucket_seq_len}"
                assert (
                    tgt_len <= bucket_seq_len
                ), f"Error: Target tokens len {tgt_len} exceeds bucket seq len {bucket_seq_len}"

                # Make sure at least one of the lengths is between the last bucket size and the
                # current one
                assert (prev_bucket_seq_len < src_len <= bucket_seq_len) or (
                    prev_bucket_seq_len < tgt_len <= bucket_seq_len
                ), (
                    "Error: Neither source nor target tokens length is in the range "
                    f"({prev_bucket_seq_len}, {bucket_seq_len}]"
                )

        print(f"Processed {batch_counter} batches.")
        print(f"Processed {pair_counter} pairs.")


if __name__ == "__main__":
    main()
