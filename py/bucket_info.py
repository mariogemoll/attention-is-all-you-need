#!/usr/bin/env python3

import argparse

from tabulate import tabulate

from buckets import get_bucket_sizes, open_buckets


def print_bucket_summary(dataset_file_path_prefix: str) -> None:
    """
    Print a summary of bucket sizes using open_buckets context manager.

    Args:
        dataset_file_path_prefix: Path to the dataset file (without .bin/.idx/.bidx suffix)
    """
    with open_buckets(dataset_file_path_prefix) as bucketed:
        step_size = bucketed.step_size
        bucketed.bucket_index_file.seek(0)
        bucket_sizes = get_bucket_sizes(bucketed.bucket_index_file)
        total_entries = sum(bucket_sizes)

    print(f"Bucket summary for {dataset_file_path_prefix}.bidx:")
    print(f"Total entries: {total_entries:,}")
    print()

    # Prepare table data
    table_data = []
    for i, count in enumerate(bucket_sizes):
        if count > 0:  # Only show non-empty buckets
            max_bucket_len = (i + 1) * step_size
            min_bucket_len = i * step_size + 1 if i > 0 else 1
            percentage = (count / total_entries * 100) if total_entries > 0 else 0
            table_data.append(
                [i, f"{min_bucket_len}-{max_bucket_len}", f"{count:,}", f"{percentage:.1f}%"]
            )

    if table_data:
        headers = ["Bucket", "Token Range", "Entries", "Percentage"]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print bucket information for a bucket index")
    parser.add_argument(
        "dataset_file_path_prefix", help="Path to the dataset file (without .bin/.idx/.bidx suffix)"
    )

    args = parser.parse_args()

    try:
        print_bucket_summary(args.dataset_file_path_prefix)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
