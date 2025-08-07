import random
import sys

import tokenizers  # type: ignore

from serialization import (
    get_bucket_sizes,
    get_entry_idx_from_bucket,
    get_entry_info_from_index,
    read_from_data_file,
)


def main() -> None:
    if len(sys.argv) < 4:
        print(f"Usage: python {sys.argv[0]} tokenizer_path dataset_file_path_prefix num_samples")
        sys.exit(1)

    tokenizer_path = sys.argv[1]
    dataset_file_path_prefix = sys.argv[2]
    num_samples = int(sys.argv[3])

    bucket_sizes = get_bucket_sizes(dataset_file_path_prefix + ".cidx")

    num_buckets = len(bucket_sizes)
    if num_buckets == 0:
        print("No buckets found in the dataset index.")
        sys.exit(1)

    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    with open(dataset_file_path_prefix + ".cidx", "rb") as bucket_file, open(
        dataset_file_path_prefix + ".idx", "rb"
    ) as index_file, open(dataset_file_path_prefix + ".bin", "rb") as data_file:

        # Get data file length
        data_file.seek(0, 2)  # Seek to end
        data_file_len = data_file.tell()
        data_file.seek(0)  # Reset to start

        for i in range(num_samples):
            print()
            print("-" * 100)
            print()
            # Choose a random bucket
            bucket_index = random.randint(0, num_buckets - 1)
            bucket_size = bucket_sizes[bucket_index]

            idx_in_bucket = random.randint(0, bucket_size - 1)
            if bucket_size == 0:
                print(f"Bucket {bucket_index} is empty, skipping sample.")
                continue

            idx = get_entry_idx_from_bucket(bucket_file, bucket_index, idx_in_bucket)

            start_pos, src_len, tg_len = get_entry_info_from_index(index_file, data_file_len, idx)

            corpus_id, original_line_number, src_tokens, tgt_tokens = read_from_data_file(
                data_file, start_pos, src_len, tg_len
            )

            print(
                f"Sample {i + 1}: Bucket {bucket_index}, index in bucket {idx_in_bucket}, "
                f"entry ID {idx}, corpus ID {corpus_id}, original line {original_line_number}"
            )
            print()
            print(tokenizer.decode(src_tokens))  # Decode source tokens for display
            print()
            print(tokenizer.decode(tgt_tokens))  # Decode target tokens for display


if __name__ == "__main__":
    main()
