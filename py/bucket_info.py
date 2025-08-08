#!/usr/bin/env python3

import argparse

from serialization import print_bucket_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Print bucket information for a chunked index")
    parser.add_argument(
        "dataset_file_path_prefix", help="Path to the dataset file (without .bin/.idx/.cidx suffix)"
    )

    args = parser.parse_args()

    try:
        chunked_index_path = args.dataset_file_path_prefix + ".cidx"
        print_bucket_summary(chunked_index_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
