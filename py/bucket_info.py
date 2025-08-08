#!/usr/bin/env python3

import argparse

from serialization import print_bucket_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Print bucket information for a bucket index")
    parser.add_argument(
        "dataset_file_path_prefix", help="Path to the dataset file (without .bin/.idx/.bidx suffix)"
    )

    args = parser.parse_args()

    try:
        bucket_index_path = args.dataset_file_path_prefix + ".bidx"
        print_bucket_summary(bucket_index_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
